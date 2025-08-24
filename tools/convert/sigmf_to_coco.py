import argparse
from selfrf.finetuning.detection.convert.sigmf_to_coco import sigmf_dir_to_coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert SigMF datasets (framed) to COCO format.")
    parser.add_argument("--sigmf_input_dir", type=str, required=True,
                        help="Directory containing SigMF meta files. "
                        "COCO output will be placed in a 'coco_dataset' subdirectory here.")
    parser.add_argument("--frame_overlap", type=float,
                        default=0.5, help="Frame overlap ratio.")
    parser.add_argument("--nfft", type=int, default=512,
                        help="NFFT for spectrogram.")
    parser.add_argument("--hop_div", type=int, default=4,
                        help="Spectrogram hop length divisor.")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["detection", "recognition",],
                        help="Mode for target transforms (detection, recognition).")

    args = parser.parse_args()

    sigmf_dir_to_coco(
        sigmf_input_dir=args.sigmf_input_dir,
        frame_overlap_ratio=args.frame_overlap,
        nfft=args.nfft,
        hop_length_div=args.hop_div,
        mode=args.mode,  # "detection" or "recognition"
    )


if __name__ == "__main__":
    main()
