"""Edge Impulse "Face detection - FOMO" export, JPEG-DCT encoded on the fly."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..DareLab_Face_DCT import hw_jpeg
from ..DareLab_Face_DCT.jpeg import ZIGZAG, encode
from .transforms import FaceFOMODCTTransform, PixelAugmentation, load_letterboxed

DEFAULT_ROOT = "/home/hzhao/ZHN/datasets/face-detection-fomo-export"
COMPONENTS = ("Y", "Cb", "Cr")
CHANNELS_PER_COMPONENT = 64
# Edge Impulse split folder -> COCO file; val and test both use the testing folder.
SPLITS = {"training": "annotations/train.json", "testing": "annotations/val.json"}
DEFAULT_ANNOTATIONS = {"train": SPLITS["training"], "val": SPLITS["testing"], "test": SPLITS["testing"]}

# ITU-T T.81 Annex K tables in natural (row-major) order.
LUMA_TABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
CHROMA_TABLE = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32)


def quant_tables(quality):
    """libjpeg jpeg_set_quality scaling in zigzag order, as generate_dct_dataset.py."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    luma, chroma = (np.clip((table * scale + 50) // 100, 1, 255)[ZIGZAG]
                    for table in (LUMA_TABLE, CHROMA_TABLE))
    return {"Y": luma.tolist(), "Cb": chroma.tolist(), "Cr": chroma.tolist()}


def export_coco_annotations(img_path=DEFAULT_ROOT):
    """Write annotations/train.json (training/) and annotations/val.json (testing/).

    Reads each folder's info.labels. Boxes keep original pixel coordinates,
    clipped to the image; boxes left without area (labelled outside the photo)
    are dropped. Image IDs are unique across both files. Returns the paths.
    """
    root = Path(img_path)
    category = {"id": 1, "name": "face"}
    image_id = annotation_id = 0
    paths = []
    for folder, relative in SPLITS.items():
        with (root / folder / "info.labels").open(encoding="utf-8") as stream:
            files = sorted(json.load(stream)["files"], key=lambda item: item["path"])
        images, annotations, dropped = [], [], 0
        for item in files:
            file_name = f"{folder}/{item['path']}"
            with Image.open(root / file_name) as image:
                width, height = image.size
            image_id += 1
            images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
            for box in item.get("boundingBoxes", []):
                if box["label"] != category["name"]:
                    raise ValueError(f"Unexpected label {box['label']!r} in {file_name}")
                x0, y0 = max(0, box["x"]), max(0, box["y"])
                x1, y1 = min(width, box["x"] + box["width"]), min(height, box["y"] + box["height"])
                if x1 <= x0 or y1 <= y0:
                    dropped += 1
                    continue
                annotation_id += 1
                annotations.append({"id": annotation_id, "image_id": image_id, "category_id": category["id"],
                                    "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                    "iscrowd": 0})
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": [category]},
                                   indent=1), encoding="utf-8")
        print(f"{path}: {len(images)} images, {len(annotations)} boxes, {dropped} outside boxes dropped")
        paths.append(path)
    return paths


class FaceFOMODCTDataset(Dataset):
    """Original photos letterboxed to input_size and encoded to 8x8 block DCT.

    The photos (2000-6000 px, mixed orientation) are resized keeping the aspect
    ratio, centred on an input_size = [W, H] canvas filled with 128, then
    encoded like DareLab_Face_DCT: JPEG quality tables (quality, 0 = no
    quantization) or, with hw_encoder = {final_rounding, subsampling}, the
    bit-exact FPGA model of hw_jpeg. Output fields, normalization and
    augmentation follow DareLabFaceDCTDataset; gt boxes and warp_matrix are
    relative to the original photo, so predictions map back to it.
    The split is the COCO file ann_path (see export_coco_annotations).
    cache_images keeps the letterboxed canvases in memory (about 0.3 MB each).
    Subclasses for other single-class datasets override CATEGORY,
    DEFAULT_IMG_PATH and DEFAULT_ANNOTATIONS.
    """

    CATEGORY = "face"
    DEFAULT_IMG_PATH = DEFAULT_ROOT
    DEFAULT_ANNOTATIONS = DEFAULT_ANNOTATIONS

    def __init__(self, img_path=None, ann_path=None, mode="train", input_size=(320, 320),
                 num_dct_coefficients=64, normalization_stats=None, augmentation=None,
                 quality=75, hw_encoder=None, rgb_path=None, cache_images=True):
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val or test")
        img_path = img_path or self.DEFAULT_IMG_PATH
        if isinstance(num_dct_coefficients, int):
            num_dct_coefficients = [num_dct_coefficients] * len(COMPONENTS)
        counts = list(num_dct_coefficients)
        if (len(counts) != len(COMPONENTS)
                or any(type(count) is not int or not 0 <= count <= 64 for count in counts)
                or not sum(counts)):
            raise ValueError("num_dct_coefficients must be an integer or [Y, Cb, Cr] "
                             "integers in [0, 64] with a positive sum")
        input_size = tuple(input_size)
        if len(input_size) != 2 or any(type(v) is not int or v <= 0 or v % 16 for v in input_size):
            raise ValueError("input_size must be [W, H], both positive multiples of 16")
        self.hw_encoder = dict(hw_encoder) if hw_encoder else None
        if self.hw_encoder:
            if set(self.hw_encoder) - {"final_rounding", "subsampling"}:
                raise ValueError("hw_encoder only takes final_rounding and subsampling")
            self.hw_encoder.setdefault("final_rounding", "half_away")
            self.hw_encoder.setdefault("subsampling", "420")
            if self.hw_encoder["final_rounding"] not in hw_jpeg.FINAL_ROUNDING:
                raise ValueError(f"final_rounding must be one of {hw_jpeg.FINAL_ROUNDING}")
            if self.hw_encoder["subsampling"] not in hw_jpeg.CHROMA_FACTOR:
                raise ValueError(f"subsampling must be one of {tuple(hw_jpeg.CHROMA_FACTOR)}")
        self.img_path = Path(img_path)
        self.ann_path = Path(ann_path) if ann_path else self.img_path / self.DEFAULT_ANNOTATIONS[mode]
        self.rgb_path = Path(rgb_path) if rgb_path else self.img_path
        self.mode = mode
        self.input_size = input_size
        self.num_dct_coefficients = dict(zip(COMPONENTS, counts))
        self.quant_tables = quant_tables(quality) if quality and not self.hw_encoder else None
        self.scale = None
        if normalization_stats is not None:
            with np.load(normalization_stats, allow_pickle=False) as stats:
                scale = np.asarray(stats["scale"], dtype=np.float32)
            if scale.shape != (len(COMPONENTS) * CHANNELS_PER_COMPONENT,) or (scale <= 0).any():
                raise ValueError("normalization_stats must hold 192 positive per-channel scales")
            selected = np.concatenate([np.arange(self.num_dct_coefficients[name]) + CHANNELS_PER_COMPONENT * index
                                       for index, name in enumerate(COMPONENTS)])
            self.scale = torch.from_numpy(scale[selected]).view(-1, 1, 1)
        self.transform = FaceFOMODCTTransform()
        self.augmentation = (PixelAugmentation(augmentation)
                             if augmentation and mode == "train" else None)

        if not self.ann_path.exists():
            raise FileNotFoundError(f"{self.ann_path} does not exist; run "
                                    "dataloaders/Face_FOMO_DCT/prepare_dataset.py first")
        with self.ann_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        categories = data["categories"]
        if len(categories) != 1 or categories[0]["name"] != self.CATEGORY:
            raise ValueError(f"{type(self).__name__} requires a single {self.CATEGORY!r} category")
        category_id = categories[0]["id"]
        self.class_names = [self.CATEGORY]
        self.data_info = sorted(data["images"], key=lambda info: info["id"])
        self.img_ids = [info["id"] for info in self.data_info]
        if len(set(self.img_ids)) != len(self.img_ids):
            raise ValueError("Image IDs must be unique")
        self.annotations = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] != category_id:
                raise ValueError("Annotation refers to an unknown foreground category")
            self.annotations[ann["image_id"]].append(ann)
        unknown = set(self.annotations) - set(self.img_ids)
        if unknown:
            raise ValueError(f"Annotations refer to unknown image IDs: {sorted(unknown)[:5]}")

        self.cache = None
        if cache_images:
            with ThreadPoolExecutor(8) as pool:
                self.cache = list(pool.map(self._load, self.data_info))

    def __len__(self):
        return len(self.data_info)

    def _load(self, info):
        canvas, matrix, size = load_letterboxed(self.rgb_path / info["file_name"], self.input_size)
        if size != (info["width"], info["height"]):
            raise ValueError(f"Image size disagrees with COCO: {info['file_name']}")
        return canvas, matrix

    def encode(self, image, counts=None):
        """uint8 RGB canvas -> {component: float32 [k, h, w]} quantized coefficients."""
        counts = counts or self.num_dct_coefficients
        if self.hw_encoder:
            return hw_jpeg.encode(image, counts, self.hw_encoder["final_rounding"],
                                  self.hw_encoder["subsampling"])
        return encode(image, counts, self.quant_tables)

    def __getitem__(self, index):
        info = self.data_info[index]
        width, height = info["width"], info["height"]
        boxes, ignored = [], []
        for ann in self.annotations[info["id"]]:
            x, y, w, h = ann["bbox"]
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            (ignored if ann.get("iscrowd", 0) or ann.get("ignore", 0) else boxes).append(box)

        image, matrix = self.cache[index] if self.cache is not None else self._load(info)
        augmentation = None
        if self.augmentation is not None:
            image, augmentation = self.augmentation(image)
            matrix = augmentation @ matrix
        sample = self.transform(self.encode(image), boxes, ignored, matrix, (width, height))
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
        sample["img_info"] = {"id": info["id"], "file_name": info["file_name"],
                              "width": width, "height": height}
        self.add_targets(sample, index, augmentation)
        return sample

    def add_targets(self, sample, index, augmentation):
        """Subclass hook for extra per-sample targets; augmentation is the canvas matrix or None."""


def compute_normalization_stats(output_path, img_path=DEFAULT_ROOT, ann_path=None, input_size=(320, 320),
                                quality=75, hw_encoder=None, lower_percentile=1.0, upper_percentile=99.0,
                                dataset_cls=None):
    """Per-channel symmetric scales on the unaugmented training split.

    Same rule and npz layout as DareLab_Face_DCT.compute_normalization_stats;
    only coefficients inside the letterboxed photo (valid_mask) are counted,
    so the 128 padding does not pull the percentiles towards 0.
    dataset_cls (default FaceFOMODCTDataset) selects the dataset subclass.
    """
    dataset = (dataset_cls or FaceFOMODCTDataset)(
        img_path=img_path, ann_path=ann_path, mode="train", input_size=input_size,
        num_dct_coefficients=CHANNELS_PER_COMPONENT, quality=quality, hw_encoder=hw_encoder)
    counts = dict.fromkeys(COMPONENTS, CHANNELS_PER_COMPONENT)
    values = [[] for _ in range(len(COMPONENTS) * CHANNELS_PER_COMPONENT)]
    for index, info in enumerate(dataset.data_info):
        image, matrix = dataset.cache[index]
        coefficients = dataset.encode(image, counts)
        # Bounding block rectangle of the photo on each component's native grid.
        corners = np.array([[0, 0], [info["width"], info["height"]]], dtype=np.float64) @ matrix[:2, :2].T
        corners += matrix[:2, 2]
        for offset, name in enumerate(COMPONENTS):
            grid_h, grid_w = coefficients[name].shape[1:]
            step_y, step_x = dataset.input_size[1] / grid_h, dataset.input_size[0] / grid_w
            x0, y0 = int(corners[0, 0] // step_x), int(corners[0, 1] // step_y)
            x1, y1 = int(np.ceil(corners[1, 0] / step_x)), int(np.ceil(corners[1, 1] / step_y))
            for channel, plane in enumerate(coefficients[name]):
                values[offset * CHANNELS_PER_COMPONENT + channel].append(plane[y0:y1, x0:x1].ravel())
    values = [np.concatenate(channel) for channel in values]
    lower = np.array([np.percentile(v, lower_percentile) for v in values], dtype=np.float32)
    upper = np.array([np.percentile(v, upper_percentile) for v in values], dtype=np.float32)
    maximum = np.array([np.abs(v).max() for v in values], dtype=np.float32)
    scale = np.maximum(np.abs(lower), np.abs(upper))
    scale = np.where(scale > 0, scale, np.where(maximum > 0, maximum, 1.0)).astype(np.float32)
    np.savez(output_path, lower=lower, upper=upper, max_abs=maximum, scale=scale,
             percentiles=np.array([lower_percentile, upper_percentile], dtype=np.float32),
             ann_path=str(dataset.ann_path), num_images=len(dataset),
             input_size=np.array(dataset.input_size), quality=quality,
             hw_encoder=json.dumps(dataset.hw_encoder))
    return scale
