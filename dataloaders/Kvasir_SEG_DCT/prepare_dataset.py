"""Split Kvasir-SEG, write COCO box annotations and compute the normalization stats.

Run directly: python Object_Detection_DCT/dataloaders/Kvasir_SEG_DCT/prepare_dataset.py.
Writes annotations/{train,val,test}.json and the npz under --root. The stats are
measured on the training split with the same encoder options as Face_FOMO_DCT.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from Object_Detection_DCT.dataloaders.Face_FOMO_DCT.dataset import compute_normalization_stats  # noqa: E402
from Object_Detection_DCT.dataloaders.Kvasir_SEG_DCT.dataset import (  # noqa: E402
    DEFAULT_ROOT, SPLIT_SEED, KvasirSEGDCTDataset, export_coco_annotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--input-size", type=int, nargs=2, default=[320, 320], metavar=("W", "H"))
    parser.add_argument("--quality", type=int, default=75, help="0 = unquantized; ignored with --hw")
    parser.add_argument("--hw", action="store_true", help="Encode with the FPGA model hw_jpeg")
    parser.add_argument("--final-rounding", default="half_away", choices=("half_away", "floor"))
    parser.add_argument("--subsampling", default="420", choices=("420", "422"))
    args = parser.parse_args()

    export_coco_annotations(args.root, args.seed)
    hw_encoder = ({"final_rounding": args.final_rounding, "subsampling": args.subsampling}
                  if args.hw else None)
    encoder = f"hw{args.subsampling}_{args.final_rounding}" if args.hw else f"q{args.quality}"
    width, height = args.input_size
    output = args.root / f"train_p01_p99_symmetric_{width}x{height}_{encoder}.npz"
    scale = compute_normalization_stats(output, img_path=args.root, input_size=(width, height),
                                        quality=args.quality, hw_encoder=hw_encoder,
                                        dataset_cls=KvasirSEGDCTDataset)
    print(f"{output}: Y {scale[:16].round().tolist()}")


if __name__ == "__main__":
    main()
