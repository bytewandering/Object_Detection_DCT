"""Convert the Edge Impulse labels to COCO and compute the normalization stats.

Run directly: python Object_Detection_DCT/dataloaders/Face_FOMO_DCT/prepare_dataset.py.
Writes annotations/train.json, annotations/val.json and the npz under --root.
The stats depend on --input-size and the encoder, so they must match the config.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from Object_Detection_DCT.dataloaders.Face_FOMO_DCT.dataset import (  # noqa: E402
    DEFAULT_ROOT, compute_normalization_stats, export_coco_annotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--input-size", type=int, nargs=2, default=[320, 320], metavar=("W", "H"))
    parser.add_argument("--quality", type=int, default=75, help="0 = unquantized; ignored with --hw")
    parser.add_argument("--hw", action="store_true", help="Encode with the FPGA model hw_jpeg")
    parser.add_argument("--final-rounding", default="half_away", choices=("half_away", "floor"))
    parser.add_argument("--subsampling", default="420", choices=("420", "422"))
    parser.add_argument("--output", type=Path, help="Default: <root>/train_p01_p99_symmetric_<W>x<H>_<encoder>.npz")
    args = parser.parse_args()

    export_coco_annotations(args.root)
    hw_encoder = ({"final_rounding": args.final_rounding, "subsampling": args.subsampling}
                  if args.hw else None)
    encoder = f"hw{args.subsampling}_{args.final_rounding}" if args.hw else f"q{args.quality}"
    width, height = args.input_size
    output = args.output or args.root / f"train_p01_p99_symmetric_{width}x{height}_{encoder}.npz"
    scale = compute_normalization_stats(output, img_path=args.root, input_size=(width, height),
                                        quality=args.quality, hw_encoder=hw_encoder)
    print(f"{output}: Y {scale[:16].round().tolist()}")
    print(json.dumps({"input_size": [width, height], "quality": args.quality, "hw_encoder": hw_encoder}))


if __name__ == "__main__":
    main()
