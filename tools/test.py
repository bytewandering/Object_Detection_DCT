"""Evaluate a DCT detector Lightning checkpoint on the test split of its saved config.

Run directly: python Object_Detection_DCT/tools/test.py --checkpoint PATH.
The test split reads dataloader.val of the checkpoint config (test uses val.json).
Detections are drawn on the matching RGB frames under --image-root.
"""

import argparse
import copy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint val batch_size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-samples", type=int, help="Debug subset; default: all test images")
    parser.add_argument("--image-every", type=int, default=50,
                        help="Save images for every Nth test sample, starting with the first (default: 50)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Display and image-level threshold; does not filter COCO AP")
    parser.add_argument("--image-root", type=Path, default=Path("/home/hzhao/ZHN/datasets/DareLab_Face"),
                        help="RGB frames with the same file_name as the DCT annotations, for drawing only")
    args = parser.parse_args()
    if not args.checkpoint.expanduser().is_file():
        parser.error("--checkpoint must be an existing Lightning checkpoint")
    for key in ("batch_size", "threads", "limit_samples", "image_every"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0, 1]")
    return args


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def restore_task(checkpoint):
    from Object_Detection_DCT.tasks.training_tasks.TrainingTask_Object_Detection_DCT import (
        TrainingTask_Object_Detection_DCT)
    from Object_Detection_DCT.utils import CfgNode
    import torch.ao.quantization as aq

    if "state_dict" not in checkpoint or "cfg" not in checkpoint.get("hyper_parameters", {}):
        raise ValueError("Expected a complete Lightning checkpoint with hyper_parameters.cfg")
    task = TrainingTask_Object_Detection_DCT(CfgNode(copy.deepcopy(checkpoint["hyper_parameters"]["cfg"])))
    # Restore Python quantization ranges/dtypes before loading scale/zero_point buffers.
    task.on_load_checkpoint(checkpoint)
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    task.eval()
    task.model.apply(aq.disable_observer)
    return task


def save_detection_image(output, image_id, original, truths, predictions, threshold):
    from PIL import ImageDraw

    canvas = original.copy()
    draw = ImageDraw.Draw(canvas)
    for box in truths:
        draw.rectangle(box["bbox_xyxy"], outline="orange" if box["ignored"] else "lime", width=2)
    for box in predictions:
        if box["score"] >= threshold:
            xy = box["bbox_xyxy"]
            draw.rectangle(xy, outline="red", width=2)
            draw.text((xy[0], max(0, xy[1] - 12)), f"face {box['score']:.3f}", fill="red")
    draw.text((4, 4), "GT: green | ignored: orange | prediction: red", fill="white",
              stroke_width=1, stroke_fill="black")
    relative = Path("images") / f"{image_id}.png"
    (output / relative).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output / relative)
    return relative.as_posix()


def evaluate(task, loader, output, device, threshold=0.3, image_every=50, image_root=None):
    import torch
    from PIL import Image
    from Object_Detection_DCT.tasks.training_tasks.centroid_metrics import box_centroids, centroid_metrics, load_ground_truth

    records, samples, losses = {}, [], []
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    image_root = Path(image_root) if image_root else dataset.img_path
    task.to(device)
    task.eval()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            batch = task.transfer_batch_to_device(cpu_batch, device, 0)
            # Same forward as validation: raw predictions give the loss, then NMS post-processing.
            preds = task.model.train_forward(batch["img"], batch)
            if not torch.isfinite(preds).all().item():
                raise ValueError(f"Nonfinite predictions in test batch {batch_index}")
            loss, loss_states = task.compute_losses(preds, batch)
            size = batch["img"].shape[0]
            losses.append((size, float(loss), {k: float(v) for k, v in loss_states.items()}))
            detections = task.model.head.post_process(preds, batch)
            if records.keys() & detections.keys():
                raise ValueError("Duplicate test image IDs")
            records.update(detections)
            for index, image_id in enumerate(batch["img_info"]["id"]):
                image_id = int(image_id)
                file_name = batch["img_info"]["file_name"][index]
                width, height = (int(batch["img_info"][key][index]) for key in ("width", "height"))
                truths = []
                for annotation in dataset.annotations[image_id]:
                    x, y, w, h = annotation["bbox"]
                    truths.append({"bbox_xyxy": [max(0, x), max(0, y), min(width, x + w), min(height, y + h)],
                                   "ignored": bool(annotation.get("iscrowd", 0) or annotation.get("ignore", 0))})
                boxes = sorted(({"bbox_xyxy": [float(v) for v in box[:4]], "score": float(box[4])}
                                for box in detections[image_id].get(0, [])),
                               key=lambda box: -box["score"])
                has_face = any(not box["ignored"] for box in truths)
                detected = any(box["score"] >= threshold for box in boxes)
                selected = len(samples) % image_every == 0
                path = None
                if selected:
                    with Image.open(image_root / file_name) as image:
                        path = save_detection_image(output, image_id, image.convert("RGB"),
                                                    truths, boxes, threshold)
                samples.append({
                    "image_id": image_id, "file_name": file_name, "width": width, "height": height,
                    "recording": Path(file_name).parts[0],
                    "has_face": has_face, "detected_at_threshold": detected,
                    "max_score": boxes[0]["score"] if boxes else 0.0,
                    "ground_truth_boxes": truths, "predicted_boxes": boxes, "image": path,
                })
            if batch_index % 10 == 0 or len(samples) == len(loader.dataset):
                print(f"Evaluated {len(samples)}/{len(loader.dataset)} images", flush=True)
    if not samples:
        raise ValueError("Test set is empty")

    summary = {key.removeprefix("test/"): value for key, value in task._coco_metrics(records, "test").items()}
    total = sum(size for size, _, _ in losses)
    summary["loss"] = sum(size * value for size, value, _ in losses) / total
    for key in losses[0][2]:
        summary[key] = sum(size * states[key] for size, _, states in losses) / total
    # Image-level face presence at the display threshold.
    tp = sum(s["has_face"] and s["detected_at_threshold"] for s in samples)
    fp = sum(not s["has_face"] and s["detected_at_threshold"] for s in samples)
    fn = sum(s["has_face"] and not s["detected_at_threshold"] for s in samples)
    tn = len(samples) - tp - fp - fn
    summary["image_level"] = {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(samples), "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }
    per_recording = {}
    for sample in samples:
        entry = per_recording.setdefault(sample["recording"], {"images": 0, "with_face": 0, "detected": 0})
        entry["images"] += 1
        entry["with_face"] += sample["has_face"]
        entry["detected"] += sample["detected_at_threshold"]
    summary["per_recording"] = per_recording
    # Same centroid matching as FOMO, using box centres, for a direct comparison.
    summary["centroid"] = centroid_metrics(
        {image_id: box_centroids(detections) for image_id, detections in records.items()},
        load_ground_truth(dataset.ann_path), threshold)
    return summary, samples


def main():
    args = parse_args()
    import torch
    from Object_Detection_DCT.dataloaders import build_dataloader

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data = checkpoint_path.read_bytes()
    task = restore_task(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
    cfg = task.cfg
    loader_cfg = copy.deepcopy(cfg.dataloader.val)
    for key in ("img_path", "ann_path"):
        if loader_cfg.get(key):
            loader_cfg[key] = str(project_path(loader_cfg[key]))
    loader_cfg.update(workers=args.workers, num_workers=args.workers, shuffle=False, drop_last=False)
    if args.workers == 0:
        loader_cfg.update(persistent_workers=False, prefetch_factor=None)
    if args.batch_size is not None:
        loader_cfg.batch_size = args.batch_size
    gpu_ids = cfg.get("device", {}).get("gpu_ids", [0])
    device = torch.device(f"cuda:{gpu_ids[0]}" if not args.cpu and torch.cuda.is_available() and gpu_ids else "cpu")
    loader = build_dataloader(loader_cfg, mode="test", pin_memory=device.type == "cuda")
    total = len(loader.dataset)
    if args.limit_samples is not None:
        subset = torch.utils.data.Subset(loader.dataset, range(min(args.limit_samples, total)))
        loader = torch.utils.data.DataLoader(subset, batch_size=loader.batch_size,
                                             collate_fn=loader.collate_fn, num_workers=args.workers,
                                             pin_memory=device.type == "cuda")
    # Save beside the run that holds the checkpoint, even if its directory was renamed.
    run_dir = (checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints"
               else project_path(cfg.save_dir))
    output = run_dir / "test results" / (
        f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint stage: {task.quantization_mode}; device: {device}; output: {output}", flush=True)
    summary, samples = evaluate(task, loader, output, device, args.score_threshold, args.image_every,
                               args.image_root.expanduser())
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        "saved_config": cfg, "quantization_mode": task.quantization_mode,
        "quantization": "checkpoint fake-quant parameters preserved; observers frozen",
        "input": "JPEG Q75 quantized DCT coefficients / 128; num_dct_coefficients = "
                 f"{loader_cfg.get('num_dct_coefficients', 64)} (Y/Cb/Cr zigzag)",
        "image_root": str(args.image_root.expanduser()),
        "annotation_file": str(loader_cfg.ann_path), "device": str(device),
        "batch_size": loader.batch_size, "test_dataset_size": total, "is_subset": len(samples) < total,
        "score_threshold": args.score_threshold,
        "image_sampling": {"every": args.image_every, "start_index": 0,
                           "saved_samples": sum(sample["image"] is not None for sample in samples)},
        "coordinate_system": "original image pixels, xyxy; class 0 = face",
        "metric_notes": {
            "detection": "COCO bbox AP over all evaluated images (max 100 detections, score >= 0.05 after NMS); "
                         "no score-threshold filtering; -1 means undefined.",
            "loss": "Detection loss (qfl + bbox + dfl) averaged per image, as in validation.",
            "image_level": "An image is predicted to contain a face if any box scores >= score_threshold; "
                           "ground truth from non-ignored boxes.",
            "centroid": "FOMO-style metric at score_threshold: a box centre inside an unmatched GT box is a "
                        "TP, otherwise FP; unmatched GT boxes are FN.",
            "images": "RGB frames from image_root; paths relative to results.json; "
                      "boxes below score_threshold are not drawn.",
        },
        "summary": summary, "samples": samples,
    }
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "results.json")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Results saved: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
