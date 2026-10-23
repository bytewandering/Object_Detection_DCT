"""List every false positive and false negative of a FOMO test result.

Run directly: python Object_Detection_DCT/tools/error_analysis.py "WORKSPACE/test results/model_best_int8".
Reads results.json written by test.py, repeats its centroid matching and writes
errors.csv (one row per error) and error_summary.json next to it.

False positives are typed by where the centroid lies: duplicate (inside an
already matched face box), near_face (within half a box size around a face),
background (elsewhere). False negatives are merged (a detected object's cell
box overlaps the face, so it was absorbed into another object) or missed (no
object overlaps it). Face size is the box width on the input canvas.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def inside(point, box, margin=0.0):
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - margin * w <= point[0] <= box[2] + margin * w
            and box[1] - margin * h <= point[1] <= box[3] + margin * h)


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", type=Path, help="test results/<checkpoint> directory holding results.json")
    args = parser.parse_args()
    report = json.loads((args.result_dir / "results.json").read_text(encoding="utf-8"))
    input_size = report["saved_config"]["dataloader"]["val"].get("input_size", [320, 320])

    rows = []
    for sample in report["samples"]:
        scale = min(input_size[0] / sample["width"], input_size[1] / sample["height"])
        faces = [box["bbox_xyxy"] for box in sample["ground_truth_boxes"] if not box["ignored"]]
        ignored = [box["bbox_xyxy"] for box in sample["ground_truth_boxes"] if box["ignored"]]
        matched = [False] * len(faces)
        base = {"image_id": sample["image_id"], "file_name": sample["file_name"],
                "image": sample["image"] or "", "faces_in_image": len(faces)}
        # Same greedy order as centroid_metrics.match_image.
        for prediction in sorted(sample["predictions"], key=lambda item: -item["score"]):
            point = prediction["centroid"]
            hit = next((i for i, box in enumerate(faces) if not matched[i] and inside(point, box)), None)
            if hit is not None:
                matched[hit] = True
                continue
            if any(inside(point, box) for box in ignored):
                continue
            kind = ("duplicate" if any(inside(point, box) for box in faces)
                    else "near_face" if any(inside(point, box, 0.5) for box in faces) else "background")
            rows.append({**base, "error": "FP", "type": kind, "score": round(prediction["score"], 4),
                         "x": round(point[0]), "y": round(point[1]),
                         "box_xyxy": [round(v) for v in prediction["bbox_xyxy"]], "face_width_input_px": ""})
        for i, box in enumerate(faces):
            if matched[i]:
                continue
            kind = "merged" if any(overlaps(p["bbox_xyxy"], box) for p in sample["predictions"]) else "missed"
            rows.append({**base, "error": "FN", "type": kind, "score": "",
                         "x": round((box[0] + box[2]) / 2), "y": round((box[1] + box[3]) / 2),
                         "box_xyxy": [round(v) for v in box],
                         "face_width_input_px": round((box[2] - box[0]) * scale, 1)})

    fields = ["image_id", "file_name", "error", "type", "score", "x", "y", "box_xyxy",
              "face_width_input_px", "faces_in_image", "image"]
    with (args.result_dir / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = report["summary"]
    widths = sorted(row["face_width_input_px"] for row in rows if row["error"] == "FN")
    per_image = Counter((row["image_id"], row["error"]) for row in rows)
    images = sorted({row["image_id"] for row in rows},
                    key=lambda i: -(per_image[(i, "FP")] + per_image[(i, "FN")]))
    result = {
        "checkpoint": report["checkpoint"], "score_threshold": report["model"]["score_threshold"],
        "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
        "fp_types": dict(Counter(row["type"] for row in rows if row["error"] == "FP")),
        "fn_types": dict(Counter(row["type"] for row in rows if row["error"] == "FN")),
        "fn_face_width_input_px": {"small_lt_32": sum(w < 32 for w in widths),
                                   "medium_32_64": sum(32 <= w < 64 for w in widths),
                                   "large_ge_64": sum(w >= 64 for w in widths)},
        "images_with_errors": len(images),
        "worst_images": [{"image_id": i, "fp": per_image[(i, "FP")], "fn": per_image[(i, "FN")]}
                         for i in images[:15]],
    }
    if (result["tp"], result["fp"], result["fn"]) != (
            summary["tp"], len([r for r in rows if r["error"] == "FP"]), len([r for r in rows if r["error"] == "FN"])):
        raise ValueError("Recomputed errors disagree with results.json")
    (args.result_dir / "error_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"Saved: {args.result_dir / 'errors.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
