"""Encode DareLab_Face with the bit-exact FPGA model (hw_jpeg) and compare with the float dataset.

Run directly: python Object_Detection_DCT/dataloaders/DareLab_Face_DCT/generate_hw_dct_dataset.py.
The output has the layout of the reference DCT dataset (uint8 PNG = q + 128,
label.json, annotations/, dct_meta.json), so tests only need a different img_path.
dct_meta.json gains hw_encoder, which makes DareLabFaceDCTDataset re-encode
augmented frames with hw_jpeg too. hw_diff_stats.json counts, per split,
component and zigzag index, how far hardware values are from the reference PNGs
(Y only for 4:2:2, whose chroma grid differs from the 4:2:0 reference).
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from Object_Detection_DCT.dataloaders.DareLab_Face_DCT import hw_jpeg  # noqa: E402
from Object_Detection_DCT.dataloaders.DareLab_Face_DCT.jpeg import ZIGZAG  # noqa: E402

COMPONENTS = ("Y", "Cb", "Cr")
COUNTS = dict.fromkeys(COMPONENTS, 64)
# Histogram bins of hardware - reference: <=-3, -2, -1, 0, 1, 2, >=3.
DIFF_BINS = ["<=-3", "-2", "-1", "0", "1", "2", ">=3"]


def convert(job):
    source, reference, destination, final_rounding, subsampling = job
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    destination.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros((len(COMPONENTS), 64, len(DIFF_BINS)), dtype=np.int64)
    for c, (name, planes) in enumerate(hw_jpeg.encode(rgb, COUNTS, final_rounding, subsampling).items()):
        values = planes + 128
        if values.min() < 0 or values.max() > 255:
            raise ValueError(f"Hardware coefficient exceeds the uint8 storage range: {source}")
        compare = name == "Y" or subsampling == "420"
        for index, plane in enumerate(values.astype(np.uint8)):
            if compare:
                stored = cv2.imread(str(reference / f"{name}_{index:02d}.png"), cv2.IMREAD_UNCHANGED)
                if stored is None or stored.shape != plane.shape:
                    raise FileNotFoundError(f"Reference coefficient missing or mismatched: {reference}")
                diff = np.clip(plane.astype(np.int64) - stored, -3, 3) + 3
                histogram[c, index] = np.bincount(diff.ravel(), minlength=len(DIFF_BINS))
            path = destination / f"{name}_{index:02d}.png"
            if not cv2.imwrite(str(path), plane):
                raise OSError(f"Failed to write {path}")
    return histogram


def split_stats(histogram, counts):
    """Summary over the first counts[name] coefficients plus per-index details."""
    zero = DIFF_BINS.index("0")
    stats = {"summary": {}, "per_index": {}}
    for c, name in enumerate(COMPONENTS):
        if not histogram[c].any():
            continue  # not compared
        selected = histogram[c, :counts[c]].sum(axis=0)
        stats["summary"][name] = {
            "first_k": counts[c], "values": int(selected.sum()),
            "mismatch_rate": float(1 - selected[zero] / max(selected.sum(), 1)),
            "diff_histogram": dict(zip(DIFF_BINS, selected.tolist())),
        }
        stats["per_index"][name] = [
            {"zigzag": index, "mismatch_rate": float(1 - row[zero] / max(row.sum(), 1)),
             "diff_histogram": row.tolist()}
            for index, row in enumerate(histogram[c])]
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("/home/hzhao/ZHN/datasets/DareLab_Face"),
                        help="RGB frames")
    parser.add_argument("--reference", type=Path, default=Path("/home/hzhao/ZHN/datasets/DareLab_Face_DCT"),
                        help="Float Q75 DCT dataset to copy annotations from and compare against")
    parser.add_argument("--dst", type=Path, default=Path("/home/hzhao/ZHN/datasets/DareLab_Face_DCT_HW"))
    parser.add_argument("--final-rounding", choices=hw_jpeg.FINAL_ROUNDING, default="half_away",
                        help="How q_coeff (12 fractional bits) becomes an integer; not in the Verilog")
    parser.add_argument("--subsampling", choices=tuple(hw_jpeg.CHROMA_FACTOR), default="420",
                        help="Chroma subsampling before the DCT; 422 averages horizontal pairs")
    parser.add_argument("--counts", type=int, nargs=3, default=[16, 4, 4], metavar=("Y", "CB", "CR"),
                        help="Leading zigzag coefficients summarized per component (model input)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    with (args.reference / "dct_meta.json").open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if meta.get("quality") != 75:
        raise ValueError("The Verilog tables are Q75; the reference dataset must be Q75 too")
    if args.dst.resolve() == args.reference.resolve():
        raise ValueError("--dst must differ from --reference")
    with (args.reference / "label.json").open(encoding="utf-8") as stream:
        images = json.load(stream)["images"]
    splits = {}
    for split in ("train", "val"):
        with (args.reference / "annotations" / f"{split}.json").open(encoding="utf-8") as stream:
            splits.update({info["file_name"]: split for info in json.load(stream)["images"]})

    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.reference / "label.json", args.dst / "label.json")
    shutil.copytree(args.reference / "annotations", args.dst / "annotations", dirs_exist_ok=True)
    frames = [Path(info["file_name"]).with_suffix("") for info in images]
    jobs = [(args.src / info["file_name"], args.reference / frame, args.dst / frame, args.final_rounding,
             args.subsampling)
            for info, frame in zip(images, frames)]
    histograms = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for count, (info, histogram) in enumerate(zip(images, pool.map(convert, jobs, chunksize=8)), 1):
            for split in ("all", splits.get(info["file_name"], "unsplit")):
                histograms[split] = histograms.get(split, 0) + histogram
            if count % 200 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)}", flush=True)

    factor_h, factor_w = hw_jpeg.CHROMA_FACTOR[args.subsampling]
    meta.update({
        "subsampling": ("4:2:0, 2x2 average (Cb/Cr)" if args.subsampling == "420"
                        else "4:2:2, horizontal pair average (a + b + 1) >> 1 (Cb/Cr)"),
        "CbCr_shape_hw": [images[0]["height"] // (8 * factor_h), images[0]["width"] // (8 * factor_w)],
        "transform": "bit-exact jpeg_dct_original_opt.v: integer row then column DCT, each >>> 9 "
                     "(floor); C ~= 2 x orthonormal DCT-II",
        "quantization": f"bit-exact jpeg_quant_zigzag.v: C * floor(4096 / (2Q)), final rounding "
                        f"{args.final_rounding} of the 12 fractional bits",
        "quant_tables": {name: table[ZIGZAG].tolist() for name, table in hw_jpeg.TABLES.items()},
        "hw_encoder": {"model": "Object_Detection_DCT/dataloaders/DareLab_Face_DCT/hw_jpeg.py",
                       "final_rounding": args.final_rounding, "subsampling": args.subsampling,
                       "dct_shift": hw_jpeg.DCT_SHIFT,
                       "q_frac": hw_jpeg.Q_FRAC, "reference": str(args.reference)},
    })
    with (args.dst / "dct_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)

    report = {"reference": str(args.reference), "final_rounding": args.final_rounding,
              "subsampling": args.subsampling,
              "diff": "hardware - reference quantized value", "diff_bins": DIFF_BINS,
              "splits": {split: split_stats(histogram, args.counts) for split, histogram in histograms.items()}}
    with (args.dst / "hw_diff_stats.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for split, stats in report["splits"].items():
        for name, entry in stats["summary"].items():
            print(f"{split:5s} {name:2s} first {entry['first_k']:2d}: mismatch {entry['mismatch_rate']:.2%}  "
                  f"{entry['diff_histogram']}")
    print(f"Saved {args.dst}")


if __name__ == "__main__":
    main()
