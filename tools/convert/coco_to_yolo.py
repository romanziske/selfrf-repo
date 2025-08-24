
import argparse
from pathlib import Path

from selfrf.finetuning.detection.convert.coco_to_yolo import convert_coco_to_yolo


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to Ultralytics YOLO format.")
    parser.add_argument("--coco_dir", type=Path, required=True,
                        help="Root directory of COCO dataset (contains 'annotations')")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], choices=[
                        "train", "val"], help="Dataset splits to convert")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink images instead of copying")
    parser.add_argument("--mode", type=str, required=True, choices=[
                        "detection", "recognition"], help="detection: single-class; recognition: keep COCO classes")
    args = parser.parse_args()

    # Call the conversion function with parsed arguments
    convert_coco_to_yolo(
        coco_dir=args.coco_dir,
        out_dir=args.out_dir,
        splits=args.splits,
        symlink=args.symlink,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
