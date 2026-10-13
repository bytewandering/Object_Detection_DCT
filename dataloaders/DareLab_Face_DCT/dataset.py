"""DareLab face dataset backed by precomputed 8x8 block DCT coefficient PNGs."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .jpeg import encode
from .transforms import DareLabFaceDCTTransform, PixelAugmentation

COMPONENTS = ("Y", "Cb", "Cr")
# Per-split COCO files; test evaluates on val.json.
DEFAULT_ANNOTATIONS = {"train": "annotations/train.json", "val": "annotations/val.json",
                       "test": "annotations/val.json"}
CHANNELS_PER_COMPONENT = 64


class DareLabFaceDCTDataset(Dataset):
    """Load DCT coefficients written by generate_dct_dataset.py.

    The split is defined by the COCO file: every image in ann_path is used.
    Without ann_path, train reads annotations/train.json and val/test read
    annotations/val.json under img_path.
    img is [kY + kCb + kCr, H/8, W/8]: the first kY, kCb and kCr zigzag
    coefficients of Y, Cb and Cr, with 4:2:0 chroma repeated 2x2 onto the luma
    block grid. num_dct_coefficients is one k for all components or [kY, kCb, kCr].
    Each coefficient channel c is normalized symmetrically, clip(q / s_c, -1, 1),
    where s_c = max(|p_low|, |p_high|) of that channel's quantized values on the
    training split (normalization_stats, see compute_normalization_stats). A
    zero coefficient therefore stays 0, so zero padding still means "no signal".
    augmentation (train mode only; keys as in PixelAugmentation) augments the
    original RGB frame under rgb_path and encodes it on the fly with the stored
    pipeline (dct_meta.json quantization tables) instead of reading the PNGs.
    """

    def __init__(self, img_path="/home/hzhao/ZHN/datasets/DareLab_Face_DCT",
                 ann_path=None, mode="train",
                 num_dct_coefficients=64, normalization_stats=None,
                 augmentation=None, rgb_path="/home/hzhao/ZHN/datasets/DareLab_Face"):
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val or test")
        if ann_path is None:
            ann_path = Path(img_path) / DEFAULT_ANNOTATIONS[mode]
        if isinstance(num_dct_coefficients, int):
            num_dct_coefficients = [num_dct_coefficients] * len(COMPONENTS)
        counts = list(num_dct_coefficients)
        if (len(counts) != len(COMPONENTS)
                or any(type(count) is not int or not 0 <= count <= 64 for count in counts)
                or not sum(counts)):
            raise ValueError("num_dct_coefficients must be an integer or [Y, Cb, Cr] "
                             "integers in [0, 64] with a positive sum")
        self.img_path = Path(img_path)
        self.ann_path = Path(ann_path)
        self.mode = mode
        self.num_dct_coefficients = dict(zip(COMPONENTS, counts))
        with (self.img_path / "dct_meta.json").open(encoding="utf-8") as stream:
            self.meta = json.load(stream)
        # Quantized datasets store q + 128 in uint8; others store round(c) + 32768.
        self.quantized = bool(self.meta.get("quantization"))
        self.offset = 128 if self.quantized else 32768
        self.scale = None
        if normalization_stats is not None:
            with np.load(normalization_stats, allow_pickle=False) as stats:
                scale = np.asarray(stats["scale"], dtype=np.float32)
            if scale.shape != (len(COMPONENTS) * CHANNELS_PER_COMPONENT,) or (scale <= 0).any():
                raise ValueError("normalization_stats must hold 192 positive per-channel scales")
            selected = np.concatenate([np.arange(self.num_dct_coefficients[name]) + CHANNELS_PER_COMPONENT * index
                                       for index, name in enumerate(COMPONENTS)])
            self.scale = torch.from_numpy(scale[selected]).view(-1, 1, 1)
        self.transform = DareLabFaceDCTTransform()
        self.augmentation = (PixelAugmentation(augmentation)
                             if augmentation and mode == "train" else None)
        self.rgb_path = Path(rgb_path)
        self.quant_tables = self.meta["quant_tables"] if self.quantized else None

        with self.ann_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        categories = data["categories"]
        if len(categories) != 1 or categories[0]["name"] != "face":
            raise ValueError("DareLabFaceDCTDataset requires a single 'face' category")
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

        self.img_ids = [info["id"] for info in self.data_info]
        self.annotations = {image_id: self.annotations.get(image_id, [])
                            for image_id in self.img_ids}

    def __len__(self):
        return len(self.data_info)

    def load_coefficients(self, info):
        """Return {component: float32 [k, h, w]} coefficients of one frame."""
        frame = self.img_path / Path(info["file_name"]).with_suffix("")
        luma_shape = (info["height"] // 8, info["width"] // 8)
        coefficients = {}
        for name in COMPONENTS:
            scale = 1 if name == "Y" else 2
            shape = (luma_shape[0] // scale, luma_shape[1] // scale)
            values = np.empty((self.num_dct_coefficients[name], *shape), dtype=np.float32)
            for index in range(len(values)):
                path = frame / f"{name}_{index:02d}.png"
                plane = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if plane is None:
                    raise FileNotFoundError(f"DCT coefficient does not exist: {path}")
                if plane.shape != shape:
                    raise ValueError(f"DCT grid disagrees with COCO size: {path}")
                values[index] = plane
            values -= self.offset
            coefficients[name] = values
        return coefficients

    def load_rgb(self, info):
        """Return the original frame as uint8 RGB, decoded as the generator did."""
        with Image.open(self.rgb_path / info["file_name"]) as source:
            image = np.asarray(source.convert("RGB"))
        if image.shape[:2] != (info["height"], info["width"]):
            raise ValueError(f"RGB frame disagrees with COCO size: {info['file_name']}")
        return image

    def __getitem__(self, index):
        info = self.data_info[index]
        width, height = info["width"], info["height"]
        boxes, ignored = [], []
        for ann in self.annotations[info["id"]]:
            x, y, w, h = ann["bbox"]
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Box lies outside image in annotation {ann['id']}")
            destination = ignored if ann.get("iscrowd", 0) or ann.get("ignore", 0) else boxes
            destination.append(box)

        if self.augmentation is None:
            sample = self.transform(self.load_coefficients(info), boxes, ignored)
        else:
            image, pixel_matrix = self.augmentation(self.load_rgb(info))
            coefficients = encode(image, self.num_dct_coefficients, self.quant_tables)
            sample = self.transform(coefficients, boxes, ignored, pixel_matrix)
        if self.scale is not None:
            sample["img"] = (sample["img"] / self.scale).clamp_(-1.0, 1.0)
        sample["reconstruction_target"] = sample["img"].clone()
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


def compute_normalization_stats(output_path, img_path="/home/hzhao/ZHN/datasets/DareLab_Face_DCT",
                                ann_path=None, lower_percentile=1.0, upper_percentile=99.0):
    """Save per-channel percentiles and symmetric scales of the training split.

    All 192 channels (Y/Cb/Cr x 64 zigzag coefficients) are measured on their native
    grids. scale = max(|lower|, |upper|); a channel whose percentiles are both 0 falls
    back to its largest absolute value, and an always-zero channel to 1.
    """
    dataset = DareLabFaceDCTDataset(img_path=img_path, ann_path=ann_path, mode="train",
                                    num_dct_coefficients=CHANNELS_PER_COMPONENT)
    values = [[] for _ in range(len(COMPONENTS) * CHANNELS_PER_COMPONENT)]
    for info in dataset.data_info:
        coefficients = dataset.load_coefficients(info)
        for offset, name in enumerate(COMPONENTS):
            for index, plane in enumerate(coefficients[name]):
                values[offset * CHANNELS_PER_COMPONENT + index].append(plane.ravel())
    values = [np.concatenate(channel) for channel in values]
    lower = np.array([np.percentile(v, lower_percentile) for v in values], dtype=np.float32)
    upper = np.array([np.percentile(v, upper_percentile) for v in values], dtype=np.float32)
    maximum = np.array([np.abs(v).max() for v in values], dtype=np.float32)
    scale = np.maximum(np.abs(lower), np.abs(upper))
    scale = np.where(scale > 0, scale, np.where(maximum > 0, maximum, 1.0)).astype(np.float32)
    np.savez(output_path, lower=lower, upper=upper, max_abs=maximum, scale=scale,
             percentiles=np.array([lower_percentile, upper_percentile], dtype=np.float32),
             ann_path=str(dataset.ann_path), num_images=len(dataset))
    return scale

