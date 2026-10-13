"""Single-foreground COCO data for FullCoder's three training tasks."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import FullCoderTransform


class FullCoderDataset(Dataset):
    """Load RGB images and a COCO JSON with exactly one foreground category.

    Category IDs on disk are mapped to detection label 0. Images without
    annotations are retained as background. input_size is (height, width).
    """

    def __init__(self, img_path, ann_path, input_size=(320, 320), mode="train",
                 flip_probability=0.5):
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val or test")
        self.img_path = Path(img_path)
        self.ann_path = Path(ann_path)
        self.mode = mode
        self.transform = FullCoderTransform(
            input_size, flip_probability if mode == "train" else 0.0
        )
        with self.ann_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        categories = data["categories"]
        if len(categories) != 1:
            raise ValueError("COCO must contain exactly one foreground category; omit background")
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
