"""FOMO training with a frozen, trained FOMO teacher (knowledge distillation).

The student is trained exactly as in TrainingTask_FOMO_DCT (progressive QAT,
validation F1, per-stage best checkpoints), with the loss

    (1 - alpha) * hard loss + alpha * T^2 * KD + sum_i beta_i * feature loss_i

KD is the per-cell KL(teacher || student) of the class softmax at temperature
T, weighted by the hard loss's cell weights (object cells x object_weight,
ignored cells 0), so background cells do not drown it. Every model layer runs on
the same 40x40 block grid, so features align cell by cell:
  - "mse": both features are standardized per channel (over batch and cells)
    and compared by MSE; a train-only 1x1 conv (adapter) maps the student
    feature to the teacher's channels when they differ (or when adapter: true).
  - "attention": each feature becomes a per-image map mean_c(f^2), L2
    normalized over cells; channel counts may differ and no adapter is used.

The teacher stays in FP32 eval mode outside the module tree, so it is neither
optimized, nor prepared for QAT, nor saved. Adapter weights are kept out of
checkpoint["state_dict"], so a student checkpoint loads like any FOMO one
(tools/test.py, threshold_sweep.py). inherit lists module prefixes whose
weights are copied from the teacher where names and shapes match (e.g. the
fusion module, stem and block_0, which do not depend on expand_ratio).

Config (in addition to TrainingTask_FOMO_DCT's):
    distillation:
      teacher_dir: <run directory with config.yml and checkpoints/>
      teacher_checkpoint: model_best_fp32.ckpt
      temperature: 2.0
      alpha: 0.5
      inherit: [fusion, backbone.0, backbone.1]
      features:            # student/teacher module names, e.g. backbone.3
        - {student: fusion, teacher: fusion, weight: 1.0}
      feature_loss: mse    # or attention
      adapter: auto        # auto: only when channel counts differ; true: always
"""

from pathlib import Path

import torch
import torch.ao.quantization as aq
import torch.nn as nn
import torch.nn.functional as F

from .quantization import prepare_model_for_qat, set_quantization_mode
from .TrainingTask_FOMO_DCT import TrainingTask_FOMO_DCT
from ...models import build_detector
from ...utils import load_config

ADAPTER_KEY = "distillation_adapters"


def load_teacher(directory, checkpoint_name):
    """The teacher FOMO of a TrainingTask_FOMO_DCT run, frozen in eval mode at its checkpoint's stage."""
    directory = Path(directory).expanduser()
    cfg = load_config(directory / "config.yml")
    checkpoint = torch.load(directory / "checkpoints" / checkpoint_name, map_location="cpu", weights_only=False)
    teacher = build_detector(cfg.model)
    # Same module structure as during training, so the state_dict keys match.
    prepare_model_for_qat(teacher, "int24")
    teacher.load_state_dict({key[len("model."):]: value for key, value in checkpoint["state_dict"].items()
                             if key.startswith("model.")})
    mode = checkpoint["quantization_mode"]
    set_quantization_mode(teacher, mode, reset_observers=False)
    teacher.apply(aq.disable_observer)
    if mode != "fp32":
        teacher.apply(aq.enable_fake_quant)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, cfg, mode


def _standardize(feature):
    mean = feature.mean(dim=(0, 2, 3), keepdim=True)
    std = feature.std(dim=(0, 2, 3), keepdim=True)
    return (feature - mean) / (std + 1e-6)


def _attention(feature):
    return F.normalize(feature.pow(2).mean(dim=1).flatten(1), dim=1)


class TrainingTask_FOMO_DCT_KD(TrainingTask_FOMO_DCT):

    def __init__(self, cfg):
        super().__init__(cfg)
        options = self.cfg.distillation
        teacher, teacher_cfg, mode = load_teacher(options.teacher_dir,
                                                  options.get("teacher_checkpoint", "model_best_fp32.ckpt"))
        # object.__setattr__ keeps the teacher out of parameters(), state_dict() and .to().
        object.__setattr__(self, "teacher", teacher)
        self.temperature = float(options.get("temperature", 2.0))
        self.alpha = float(options.get("alpha", 0.5))
        self.feature_loss = options.get("feature_loss", "mse")
        if self.temperature <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError("distillation.temperature must be positive and alpha in [0, 1]")
        if self.feature_loss not in ("mse", "attention"):
            raise ValueError("distillation.feature_loss must be 'mse' or 'attention'")
        if teacher_cfg.model.get("input_channels") != self.cfg.model.get("input_channels"):
            raise ValueError("Teacher and student must read the same DCT channels")

        self.features = [dict(item) for item in options.get("features", [])]
        self._student_features, self._teacher_features = {}, {}
        channels = self._register_feature_hooks()
        adapter = options.get("adapter", "auto")
        if adapter not in ("auto", True, False):
            raise ValueError("distillation.adapter must be auto, true or false")
        adapters = {}
        for index, (student_channels, teacher_channels) in enumerate(channels):
            if self.feature_loss == "attention":
                continue
            if student_channels != teacher_channels and adapter is False:
                raise ValueError(f"Feature {self.features[index]} has {student_channels} student and "
                                 f"{teacher_channels} teacher channels; it needs an adapter")
            if adapter is True or student_channels != teacher_channels:
                adapters[str(index)] = nn.Conv2d(student_channels, teacher_channels, 1)
        # Not under self.model: the deployed student never contains them.
        self.distillation_adapters = nn.ModuleDict(adapters)

        inherited = self._inherit(options.get("inherit", []))
        self.info(f"Teacher {options.teacher_dir} ({mode}); inherited {inherited} tensors; "
                  f"features {self.features}; adapters {list(adapters)}")

    @staticmethod
    def _validate_model(cfg):
        TrainingTask_FOMO_DCT._validate_model(cfg)
        if not cfg.get("distillation", {}).get("teacher_dir"):
            raise ValueError("TrainingTask_FOMO_DCT_KD requires distillation.teacher_dir")

    def _register_feature_hooks(self):
        """Forward hooks storing each listed feature; returns (student, teacher) channel counts."""
        student_modules, teacher_modules = dict(self.model.named_modules()), dict(self.teacher.named_modules())
        for index, item in enumerate(self.features):
            for side, modules, store in (("student", student_modules, self._student_features),
                                         ("teacher", teacher_modules, self._teacher_features)):
                if item[side] not in modules:
                    raise ValueError(f"distillation.features: {side} has no module {item[side]!r}")
                modules[item[side]].register_forward_hook(
                    lambda module, inputs, output, store=store, index=index: store.__setitem__(index, output))
        if not self.features:
            return []
        probe = torch.zeros(1, self.cfg.model.input_channels, 40, 40)
        with torch.no_grad():
            training = self.model.training
            self.model.eval()
            self.model.train_forward(probe)
            self.model.train(training)
            self.teacher.train_forward(probe)
        channels = [(self._student_features[i].shape[1], self._teacher_features[i].shape[1])
                    for i in range(len(self.features))]
        self._student_features.clear()
        self._teacher_features.clear()
        return channels

    def _inherit(self, prefixes):
        """Copy teacher weights and BN statistics (not quantization state) where names and shapes match."""
        teacher_state = self.teacher.state_dict()
        student_state = self.model.state_dict()
        copied = 0
        for key, value in teacher_state.items():
            if not any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                continue
            if "fake_quant" in key or "activation_post_process" in key:
                continue
            if key in student_state and student_state[key].shape == value.shape:
                student_state[key].copy_(value)
                copied += 1
        return copied

    def trainable_parameters(self):
        return [*self.model.parameters(), *self.distillation_adapters.parameters()]

    def compute_losses(self, preds, batch):
        hard, states = self.model.loss(preds, batch)
        teacher = self.teacher
        if next(teacher.parameters()).device != preds.device:
            teacher.to(preds.device)
        with torch.no_grad():
            teacher_logits = teacher.train_forward(batch["img"], batch)
        _, weights = self.model.targets(preds, batch)
        t = self.temperature
        kl = F.kl_div(F.log_softmax(preds.float() / t, dim=1), F.log_softmax(teacher_logits.float() / t, dim=1),
                      reduction="none", log_target=True).sum(dim=1)
        loss_kd = (kl * weights).sum() / weights.numel() * t * t
        loss = (1 - self.alpha) * hard + self.alpha * loss_kd
        states = {**states, "loss_kd": loss_kd}

        for index, item in enumerate(self.features):
            student, target = self._student_features.pop(index), self._teacher_features.pop(index).detach()
            if self.feature_loss == "attention":
                value = (_attention(student.float()) - _attention(target.float())).pow(2).sum(dim=1).mean()
            else:
                if str(index) in self.distillation_adapters:
                    student = self.distillation_adapters[str(index)](student)
                value = F.mse_loss(_standardize(student.float()), _standardize(target.float()))
            states[f"loss_feat_{item['student']}"] = value
            loss = loss + float(item.get("weight", 1.0)) * value
        return loss, states

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        state = checkpoint["state_dict"]
        prefix = ADAPTER_KEY + "."
        checkpoint[ADAPTER_KEY] = {key[len(prefix):]: state.pop(key) for key in list(state) if key.startswith(prefix)}

    def on_load_checkpoint(self, checkpoint):
        # Lightning loads state_dict after this hook: put the adapters back for a resumed run.
        for key, value in checkpoint.pop(ADAPTER_KEY, {}).items():
            checkpoint["state_dict"][f"{ADAPTER_KEY}.{key}"] = value
        super().on_load_checkpoint(checkpoint)
