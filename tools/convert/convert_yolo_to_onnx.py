import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx


def convert_yolo_to_onnx(
    model_path: str,
    output_dir: str = None,
    imgsz: int = 512,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 12,
    half: bool = False,
    int8: bool = False,
    device: str = "cpu"
):
    """
    Convert YOLO model to ONNX format.

    Args:
        model_path: Path to YOLO .pt model file
        output_dir: Output directory for ONNX file (default: same as model)
        imgsz: Input image size for ONNX model
        dynamic: Enable dynamic input shapes
        simplify: Simplify ONNX model
        opset: ONNX opset version
        half: Export in FP16 precision
        int8: Export in INT8 precision
        device: Device to use for conversion
    """

    # Load YOLO model
    model = YOLO(model_path)
    print(f"Loaded YOLO model from: {model_path}")

    # Get model info
    model_info = model.info()
    print(f"Model: {model_info}")

    # Determine output path
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_path.stem}.onnx"

    print(f"Converting to ONNX...")
    print(f"Input size: {imgsz}x{imgsz}")
    print(f"Output file: {output_file}")
    print(f"Dynamic shapes: {dynamic}")
    print(f"Simplify: {simplify}")
    print(f"ONNX opset: {opset}")
    print(f"Half precision: {half}")
    print(f"INT8 quantization: {int8}")

    # Export to ONNX
    try:
        export_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            half=half,
            int8=int8,
            device=device,
            nms=True,  # Disable NMS for export
        )

        print(f"\nSuccessfully converted to ONNX!")
        print(f"ONNX model saved at: {export_path}")

        # Get file size
        file_size = Path(export_path).stat().st_size / (1024 * 1024)  # MB
        print(f"📊 File size: {file_size:.2f} MB")

        # Verify the ONNX model
        try:

            onnx_model = onnx.load(export_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed!")

            # Print model info
            print(f"📋 ONNX Model Info:")
            print(f"   Inputs: {len(onnx_model.graph.input)}")
            print(f"   Outputs: {len(onnx_model.graph.output)}")

            for inp in onnx_model.graph.input:
                shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                print(f"   Input '{inp.name}': {shape}")

            for out in onnx_model.graph.output:
                shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
                print(f"   Output '{out.name}': {shape}")

        except ImportError:
            print("Install 'onnx' package to verify the exported model")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")

        return export_path

    except Exception as e:
        print(f"Conversion failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO model to ONNX format")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO .pt model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as model directory)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Input image size for ONNX model (default: 512)"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX model simplification"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 precision"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export in INT8 precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "0", "1"],
        help="Device to use for conversion"
    )

    args = parser.parse_args()

    # Validate model file exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Convert model
    convert_yolo_to_onnx(
        model_path=args.model,
        output_dir=args.output,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        opset=args.opset,
        half=args.half,
        int8=args.int8,
        device=args.device
    )


if __name__ == "__main__":
    main()
