"""Kvasir-SEG polyp boxes, letterboxed and JPEG-DCT encoded exactly as Face_FOMO_DCT."""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..Face_FOMO_DCT.dataset import FaceFOMODCTDataset
from ..Face_FOMO_DCT.transforms import BLOCK

DEFAULT_ROOT = "/home/hzhao/ZHN/datasets/Kvasir-SEG"
DEFAULT_ANNOTATIONS = {"train": "annotations/train.json", "val": "annotations/val.json",
                       "test": "annotations/test.json"}
SPLIT_SIZES = {"train": 800, "val": 100, "test": 100}
SPLIT_SEED = 0


def export_coco_annotations(img_path=DEFAULT_ROOT, seed=SPLIT_SEED):
    """Write annotations/{train,val,test}.json from kavsir_bboxes.json.

    Images are shuffled with random.Random(seed) and split 800/100/100; image
    IDs follow the sorted file names and are unique across the three files.
    Boxes keep original pixel coordinates (xmin, ymin, xmax, ymax -> COCO xywh).
    """
    root = Path(img_path)
    with (root / "kavsir_bboxes.json").open(encoding="utf-8") as stream:
        labels = json.load(stream)
    names = sorted(labels)
    if len(names) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"Expected {sum(SPLIT_SIZES.values())} images, found {len(names)}")
    ids = {name: index + 1 for index, name in enumerate(names)}
    order = names[:]
    random.Random(seed).shuffle(order)
    category = {"id": 1, "name": "polyp"}
    annotation_id, start, paths = 0, 0, []
    for split, size in SPLIT_SIZES.items():
        images, annotations = [], []
        for name in sorted(order[start:start + size], key=ids.get):
            entry = labels[name]
            file_name = f"images/{name}.jpg"
            with Image.open(root / file_name) as image:
                if image.size != (entry["width"], entry["height"]):
                    raise ValueError(f"Image size disagrees with kavsir_bboxes.json: {file_name}")
            images.append({"id": ids[name], "file_name": file_name,
                           "width": entry["width"], "height": entry["height"]})
            for box in entry["bbox"]:
                if box["label"] != category["name"]:
                    raise ValueError(f"Unexpected label {box['label']!r} in {file_name}")
                w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
                if w <= 0 or h <= 0:
                    raise ValueError(f"Empty box in {file_name}")
                annotation_id += 1
                annotations.append({"id": annotation_id, "image_id": ids[name], "category_id": category["id"],
                                    "bbox": [box["xmin"], box["ymin"], w, h], "area": w * h, "iscrowd": 0})
        start += size
        path = root / DEFAULT_ANNOTATIONS[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": [category]},
                                   indent=1), encoding="utf-8")
        print(f"{path}: {len(images)} images, {len(annotations)} boxes")
        paths.append(path)
    return paths


class KvasirSEGDCTDataset(FaceFOMODCTDataset):
    """FaceFOMODCTDataset for the single 'polyp' class; test reads test.json.

    Each sample also carries gt_mask [1, H/8, W/8]: the fraction of every 8x8
    block covered by the polyp mask (masks/<name>.jpg, foreground > 127), after
    the same letterbox and geometric augmentation as the image. It feeds the
    FOMO label_mode "mask"; boxes are kept for evaluation.
    """

    CATEGORY = "polyp"
    DEFAULT_IMG_PATH = DEFAULT_ROOT
    DEFAULT_ANNOTATIONS = DEFAULT_ANNOTATIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_cache = None
        if self.cache is not None:
            with ThreadPoolExecutor(8) as pool:
                self.mask_cache = list(pool.map(self._load_mask, range(len(self))))

    def _load_mask(self, index):
        """Letterboxed uint8 polyp mask on the input canvas, placed like the image."""
        info = self.data_info[index]
        _, matrix = self.cache[index] if self.cache is not None else self._load(info)
        path = self.img_path / "masks" / Path(info["file_name"]).name
        with Image.open(path) as source:
            if source.size != (info["width"], info["height"]):
                raise ValueError(f"Mask size disagrees with COCO: {path}")
            mask = (np.asarray(source.convert("L")) > 127).astype(np.float32)
        new_w, new_h = round(matrix[0, 0] * info["width"]), round(matrix[1, 1] * info["height"])
        offset_x, offset_y = int(matrix[0, 2]), int(matrix[1, 2])
        canvas = np.zeros(self.input_size[::-1], dtype=np.uint8)
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.round(resized * 255)
        return canvas

    def add_targets(self, sample, index, augmentation):
        mask = self.mask_cache[index] if self.mask_cache is not None else self._load_mask(index)
        mask = mask.astype(np.float32) / 255
        if augmentation is not None:
            # Same continuous-to-pixel convention as PixelAugmentation's cv2.warpAffine.
            half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
            pixel_matrix = np.linalg.inv(half) @ augmentation @ half
            mask = cv2.warpAffine(mask, pixel_matrix[:2], self.input_size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        width, height = self.input_size
        coverage = cv2.resize(mask, (width // BLOCK, height // BLOCK), interpolation=cv2.INTER_AREA)
        sample["gt_mask"] = torch.from_numpy(np.ascontiguousarray(coverage, dtype=np.float32))[None]
