"""Centroid precision/recall/F1 of FOMO checkpoints over a range of score thresholds.

Run directly: python Object_Detection_DCT/tools/threshold_sweep.py WORKSPACE [WORKSPACE ...].
Each checkpoint runs once on dataloader.val of its config.yml (as test.py); the
logits are kept and post_process is redone at every threshold, so objects are
re-grouped from the cells above that threshold rather than filtered by score.
Results go to WORKSPACE/threshold_sweep.json and print as one table per checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

import pytorch_lightning as pl
import torch

from Object_Detection_DCT.dataloaders import build_dataloader
from Object_Detection_DCT.tasks.testing_tasks.TestingTask_FOMO_DCT import TestingTask_FOMO_DCT
from Object_Detection_DCT.tasks.training_tasks.centroid_metrics import centroid_metrics, load_ground_truth
from Object_Detection_DCT.tools.test import find_checkpoints, project_path
from Object_Detection_DCT.utils import load_config

METRICS = ("precision", "recall", "f1", "tp", "fp", "fn", "image_precision", "image_recall")


class ThresholdSweepTask(TestingTask_FOMO_DCT):
    """Collect logits on the test split, then score them at every threshold."""

    def __init__(self, cfg, checkpoint, thresholds):
        super().__init__(cfg, checkpoint)
        self.thresholds = thresholds
        self.batches = []
        self.sweep = None

    def on_test_epoch_start(self):
        self.batches.clear()
        self._truth = load_ground_truth(self.annotation_path)

    def test_step(self, batch, batch_idx):
        logits = self.model.train_forward(batch["img"], batch)
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"Nonfinite logits in test batch {batch_idx}")
        self.batches.append((logits.cpu(), {"img_info": batch["img_info"], "warp_matrix": batch["warp_matrix"]}))

    def on_test_epoch_end(self):
        self.sweep = []
        for threshold in self.thresholds:
            self.model.score_threshold = threshold
            predictions = {}
            for logits, meta in self.batches:
                predictions.update({image_id: detections[0] for image_id, detections
                                    in self.model.post_process(logits, meta).items()})
            if len(predictions) != len(self._truth):
                raise ValueError("Every test image must be evaluated exactly once")
            metrics = centroid_metrics(predictions, self._truth, threshold, group_gap=self.group_gap)
            self.sweep.append({"threshold": threshold, **{key: metrics[key] for key in METRICS}})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspaces", nargs="+", type=Path)
    parser.add_argument("--checkpoint", nargs="+", type=Path,
                        help="Checkpoint names in each checkpoint dir; default: model_best_*.ckpt")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[round(value, 2) for value in np.arange(0.1, 0.951, 0.05)])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for workspace in args.workspaces:
        if not (workspace / "config.yml").is_file():
            parser.error(f"{workspace} must contain the config.yml written by train.py")
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("--thresholds must be in (0, 1)")
    return args


def main():
    args = parse_args()
    pl.seed_everything(42)
    for workspace in args.workspaces:
        workspace = workspace.expanduser().resolve()
        cfg = load_config(workspace / "config.yml")
        cfg.save_dir = str(workspace)
        if cfg.model.get("name") not in ("FOMO", "FOMO_JPEGFusion"):
            raise ValueError(f"{workspace} is not a FOMO run")
        loader_cfg = cfg.dataloader.val
        for key in ("img_path", "ann_path", "normalization_stats"):
            if loader_cfg.get(key):
                loader_cfg[key] = project_path(loader_cfg[key])
        loader_cfg.update(workers=0, num_workers=0, shuffle=False, drop_last=False,
                          persistent_workers=False, prefetch_factor=None)
        gpu_ids = list(cfg.get("device", {}).get("gpu_ids", [0]))
        use_gpu = torch.cuda.is_available() and not args.cpu and bool(gpu_ids)
        loader = build_dataloader(loader_cfg, mode="test", pin_memory=use_gpu)

        results = {}
        for checkpoint in find_checkpoints(cfg, args.checkpoint):
            task = ThresholdSweepTask(cfg, checkpoint, args.thresholds)
            trainer = pl.Trainer(accelerator="gpu" if use_gpu else "cpu", devices=gpu_ids[:1] if use_gpu else 1,
                                 precision="32-true", logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False,
                                 default_root_dir=cfg.save_dir)
            trainer.test(task, loader, ckpt_path=str(checkpoint), verbose=False, weights_only=True)
            results[checkpoint.name] = task.sweep
            best = max(task.sweep, key=lambda row: row["f1"])
            print(f"\n{workspace.name} / {checkpoint.name} (best F1 {best['f1']:.3f} at {best['threshold']})")
            print("thr    P      R      F1     tp   fp   fn")
            for row in task.sweep:
                print(f"{row['threshold']:.2f}  {row['precision']:.3f}  {row['recall']:.3f}  {row['f1']:.3f}  "
                      f"{row['tp']:4.0f} {row['fp']:4.0f} {row['fn']:4.0f}")
        output = workspace / "threshold_sweep.json"
        output.write_text(json.dumps({"annotation_file": str(task.annotation_path), "results": results},
                                     indent=1), encoding="utf-8")
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
