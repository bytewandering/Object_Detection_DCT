"""Standalone DareLab face dataset for classification, detection and reconstruction."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import DareLabFaceTransform


class DareLabFaceDataset(Dataset):
    """Load face COCO annotations, optionally selecting recording directories.

    Within each recording, every fifth image is held out for val/test.
    sequences optionally restricts recordings before this deterministic split.
    Classification targets are derived from valid boxes, not has_face or score.
    """

    def __init__(self, img_path="/home/hzhao/ZHN/datasets/DareLab_Face",
                 ann_path=None, input_size=(320, 240), mode="train",
                 flip_probability=0.5, sequences=None):
        if ann_path is None:
            ann_path = Path(img_path) / "label.json"
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val or test")
        self.img_path = Path(img_path)
        self.ann_path = Path(ann_path)
        self.mode = mode
        self.transform = DareLabFaceTransform(
            input_size, flip_probability if mode == "train" else 0.0
        )
        with self.ann_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        categories = data["categories"]
        if len(categories) != 1 or categories[0]["name"] != "face":
            raise ValueError("DareLabFaceDataset requires a single 'face' category")
        category_id = categories[0]["id"]
        self.cat_ids = [category_id]
        self.cat2label = {category_id: 0}
        self.class_names = [categories[0]["name"]]
        self.data_info = sorted(data["images"], key=lambda info: info["id"])
        self.img_ids = [info["id"] for info in self.data_info]
        if any(type(image_id) is not int for image_id in self.img_ids):
            raise ValueError("Image IDs must be integers")
        if len(set(self.img_ids)) != len(self.img_ids):
            raise ValueError("Image IDs must be unique")
        image_ids = set(self.img_ids)
        self.annotations = defaultdict(list)
        annotation_ids = set()
        for ann in data["annotations"]:
            if ann["id"] in annotation_ids:
                raise ValueError("Annotation IDs must be unique")
            annotation_ids.add(ann["id"])
            if ann["image_id"] not in image_ids:
                raise ValueError("Annotation refers to an unknown image ID")
            if ann["category_id"] != category_id:
                raise ValueError("Annotation refers to an unknown foreground category")
            bbox = np.asarray(ann["bbox"], dtype=np.float32)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= 0).any():
                raise ValueError(f"Invalid COCO xywh bbox in annotation {ann['id']}")
            self.annotations[ann["image_id"]].append(ann)

        if sequences is not None:
            if isinstance(sequences, str):
                sequences = [sequences]
            selected = set(sequences)
            available = {Path(info["file_name"]).parts[0] for info in self.data_info}
            if not selected or not selected <= available:
                raise ValueError(
                    f"sequences must select existing recordings: {sorted(available)}"
                )
            self.data_info = [info for info in self.data_info
                              if Path(info["file_name"]).parts[0] in selected]
        # Count separately within each recording, in the image-ID order above.
        # No random state is involved, so independently built splits agree.
        positions = defaultdict(int)
        split_info = []
        for info in self.data_info:
            recording = str(Path(info["file_name"]).parent)
            positions[recording] += 1
            is_test = positions[recording] % 5 == 0
            if is_test == (mode != "train"):
                split_info.append(info)
        self.data_info = split_info
        self.img_ids = [info["id"] for info in self.data_info]
        self.annotations = {image_id: self.annotations.get(image_id, [])
                            for image_id in self.img_ids}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        info = self.data_info[index]
        with Image.open(self.img_path / info["file_name"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        if (width, height) != (info["width"], info["height"]):
            raise ValueError(f"Image dimensions disagree with COCO: {info['file_name']}")

        boxes, ignored = [], []
        for ann in self.annotations[info["id"]]:
            x, y, w, h = ann["bbox"]
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Box lies outside image in annotation {ann['id']}")
            destination = ignored if ann.get("iscrowd", 0) or ann.get("ignore", 0) else boxes
            destination.append(box)

        sample = self.transform(image, boxes, ignored)
        count = len(sample["gt_bboxes"])
        sample["gt_labels"] = np.zeros(count, dtype=np.int64)
        sample["classification_target"] = torch.tensor([float(count > 0)])
        # An ignored-only image is not a confirmed classification negative.
        sample["classification_weight"] = torch.tensor([
            float(count > 0 or len(sample["gt_bboxes_ignore"]) == 0)
        ])
        sample["img_info"] = {
            "id": info["id"], "file_name": info["file_name"],
            "width": width, "height": height,
        }
        return sample
