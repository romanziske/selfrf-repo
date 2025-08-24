import argparse
from pathlib import Path
from selfrf.finetuning.detection.convert.torchsig_to_coco import torchsig_to_coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert TorchSig datasets to COCO format. ")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="COCO output will be placed in a 'coco' subdirectory here.")
    parser.add_argument("--nfft", type=int, default=512,
                        help="NFFT for spectrogram.")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["detection", "recognition",
                                 "family_recognition"],
                        help="Mode for target transforms (detection, recognition, family_recognition).")
    args = parser.parse_args()
    torchsig_to_coco(
        Path(args.input_dir),
        args.nfft,
        args.mode
    )


if __name__ == "__main__":
    main()
