"""Centroid detection metrics shared by FOMO and box detectors.

A prediction is a true positive when its centroid lies inside a ground-truth
box that no higher-scoring prediction in the same image has matched. Boxes
marked iscrowd/ignore are not counted as misses, and centroids landing inside
them are neither true nor false positives.

With group_gap (a fraction, e.g. 0.25), boxes that touch or lie within
group_gap x the smaller box side of each other (transitively) form one target,
matched by the box enclosing them: crowded objects count once, found or missed.
"""

import json
from pathlib import Path


def load_ground_truth(annotation_path):
    """{image_id: {"boxes": [xyxy], "ignored": [xyxy]}} for every annotated image."""
    with Path(annotation_path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    truth = {info["id"]: {"boxes": [], "ignored": []} for info in data["images"]}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        key = "ignored" if ann.get("iscrowd", 0) or ann.get("ignore", 0) else "boxes"
        truth[ann["image_id"]][key].append([x, y, x + w, y + h])
    return truth


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def group_boxes(boxes, gap):
    """Merge boxes closer than gap x the smaller side (transitively); returns the enclosing boxes."""
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            margin = gap * min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
            if (a[0] <= b[2] + margin and b[0] <= a[2] + margin
                    and a[1] <= b[3] + margin and b[1] <= a[3] + margin):
                parent[root(i)] = root(j)
    groups = {}
    for i, box in enumerate(boxes):
        groups.setdefault(root(i), []).append(box)
    return [[min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g)]
            for g in groups.values()]


def match_image(predictions, boxes, ignored, group_gap=None):
    """Greedy centroid-in-box matching; returns (tp, fp, fn). See the module doc for group_gap."""
    if group_gap is not None:
        boxes = group_boxes(boxes, group_gap)
    matched = [False] * len(boxes)
    tp = fp = 0
    for prediction in sorted(predictions, key=lambda item: -item["score"]):
        point = prediction["centroid"]
        hit = next((i for i, box in enumerate(boxes) if not matched[i] and _inside(point, box)), None)
        if hit is not None:
            matched[hit] = True
            tp += 1
        elif not any(_inside(point, box) for box in ignored):
            fp += 1
    return tp, fp, matched.count(False)


def centroid_metrics(records, truth, threshold, prefix="", group_gap=None):
    """Object- and image-level precision/recall/F1 for predictions scoring >= threshold.

    records maps image_id to a list of {"centroid": [x, y], "score": s}.
    group_gap merges crowded ground-truth boxes into one target (see the module doc).
    """
    tp = fp = fn = 0
    image = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for image_id, predictions in records.items():
        gt = truth[int(image_id)]
        kept = [p for p in predictions if p["score"] >= threshold]
        counts = match_image(kept, gt["boxes"], gt["ignored"], group_gap)
        tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
        has_face, detected = bool(gt["boxes"]), bool(kept)
        image["tp" if has_face and detected else "fp" if detected else "fn" if has_face else "tn"] += 1
    images = len(records)
    return {
        f"{prefix}precision": tp / max(tp + fp, 1), f"{prefix}recall": tp / max(tp + fn, 1),
        f"{prefix}f1": 2 * tp / max(2 * tp + fp + fn, 1),
        f"{prefix}tp": float(tp), f"{prefix}fp": float(fp), f"{prefix}fn": float(fn),
        f"{prefix}image_accuracy": (image["tp"] + image["tn"]) / max(images, 1),
        f"{prefix}image_precision": image["tp"] / max(image["tp"] + image["fp"], 1),
        f"{prefix}image_recall": image["tp"] / max(image["tp"] + image["fn"], 1),
        f"{prefix}num_images": float(images),
    }


def box_centroids(detections):
    """Convert NanoDet {class: [[x1, y1, x2, y2, score], ...]} to centroid records."""
    return [{"centroid": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], "score": float(box[4])}
            for box in detections.get(0, [])]
