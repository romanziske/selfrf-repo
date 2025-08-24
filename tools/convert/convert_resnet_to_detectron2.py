import argparse
import torch
import os
import sys
from typing import Dict, Any, List, Optional

# Define constants for commonly skipped key patterns
SKIPPED_SUFFIXES = ('.num_batches_tracked',)
SKIPPED_PREFIXES = ('fc.', 'head.', 'class_token', 'pos_embed')


def _map_key(key: str) -> str:
    """Maps a single key from timm/lightning format to Detectron2 format."""
    new_key = key

    # --- Stem Mapping ---
    if key.startswith('conv1.'):
        new_key = key.replace('conv1.', 'stem.conv1.', 1)
    elif key.startswith('bn1.'):
        new_key = key.replace('bn1.', 'stem.conv1.norm.', 1)

    # --- Layer Mapping (layerX -> resY) ---
    elif key.startswith('layer1.'):
        new_key = key.replace('layer1.', 'res2.', 1)
    elif key.startswith('layer2.'):
        new_key = key.replace('layer2.', 'res3.', 1)
    elif key.startswith('layer3.'):
        new_key = key.replace('layer3.', 'res4.', 1)
    elif key.startswith('layer4.'):
        new_key = key.replace('layer4.', 'res5.', 1)

    # --- Block-Level Mapping (within resX) ---
    # Apply these *after* layer mapping
    # Shortcut / Downsample Mapping
    if 'downsample.0.' in new_key:  # Conv in shortcut
        new_key = new_key.replace('downsample.0.', 'shortcut.', 1)
    if 'downsample.1.' in new_key:  # Norm in shortcut
        new_key = new_key.replace('downsample.1.', 'shortcut.norm.', 1)

    # Conv/Norm Mapping within Bottleneck Blocks
    new_key = new_key.replace('.bn1.', '.conv1.norm.', 1)
    new_key = new_key.replace('.bn2.', '.conv2.norm.', 1)
    new_key = new_key.replace('.bn3.', '.conv3.norm.',
                              1)  # For Bottleneck blocks

    # Prepend backbone.bottom_up. prefix for Detectron2 compatibility
    new_key = f"backbone.bottom_up.{new_key}"

    return new_key


def convert_keys(
    lightning_sd: Dict[str, torch.Tensor],
    prefix_to_remove: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Converts state_dict keys from a timm/PyTorch Lightning format to Detectron2 format.

    Handles prefix removal, skips auxiliary keys (like `num_batches_tracked`),
    skips non-backbone layers (like FC/head), and maps layer/block names.

    Args:
        lightning_sd: The state dictionary loaded from the checkpoint.
        prefix_to_remove: Optional prefix string to remove from keys (e.g., "model.backbone.").
                          If None or empty, no prefix removal is attempted.

    Returns:
        A new state dictionary with keys mapped to Detectron2 format.

    Raises:
        ValueError: If `prefix_to_remove` is specified but not found on any key.
    """
    new_sd: Dict[str, torch.Tensor] = {}
    skipped_keys: List[str] = []
    # Assume found if no prefix needed
    prefix_found_on_any_key = False if prefix_to_remove else True

    print(f"Processing {len(lightning_sd)} keys...")

    for original_key, tensor_val in lightning_sd.items():
        key_after_prefix = original_key

        # --- 1. Handle Prefix ---
        if prefix_to_remove:
            if original_key.startswith(prefix_to_remove):
                key_after_prefix = original_key[len(prefix_to_remove):]
                prefix_found_on_any_key = True
            else:
                # If prefix is mandatory, skip keys that don't have it
                # print(f"Info: Skipping key '{original_key}' because it does not start with the specified prefix '{prefix_to_remove}'.")
                skipped_keys.append(original_key)
                continue

        # --- 2. Skip Auxiliary and Non-Backbone Keys ---
        if key_after_prefix.endswith(SKIPPED_SUFFIXES):
            skipped_keys.append(original_key)
            continue
        if key_after_prefix.startswith(SKIPPED_PREFIXES):
            skipped_keys.append(original_key)
            continue

        # --- 3. Map Keys ---
        mapped_key = _map_key(key_after_prefix)

        # --- 4. Check if Mapping Occurred (Optional Warning) ---
        if mapped_key == key_after_prefix and not mapped_key.startswith('stem.'):
            # This might indicate an unhandled layer type or naming convention
            print(
                f"Warning: Key '{original_key}' (processed as '{key_after_prefix}') might not have been fully mapped. Result: '{mapped_key}'")

        new_sd[mapped_key] = tensor_val.cpu()  # Ensure tensors are on CPU

    # --- 5. Final Prefix Check ---
    if prefix_to_remove and not prefix_found_on_any_key:
        error_msg = (
            f"\nError: Specified prefix '{prefix_to_remove}' was not found on ANY keys "
            f"in the state dictionary.\nPlease check the --prefix argument or inspect "
            f"the checkpoint keys.\nFirst few keys found: {list(lightning_sd.keys())[:5]}"
        )
        raise ValueError(error_msg)

    # --- 6. Report Skipped Keys ---
    if skipped_keys:
        print(
            f"\nSkipped {len(skipped_keys)} keys (e.g., non-backbone, auxiliary, prefix mismatch):")
        limit = 20
        for i, sk in enumerate(skipped_keys):
            print(f"- {sk}")
            if i >= limit - 1 and len(skipped_keys) > limit:
                print(f"... and {len(skipped_keys) - limit} more.")
                break

    print(f"\nSuccessfully converted {len(new_sd)} keys.")
    return new_sd


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads the state dictionary from a checkpoint file.

    Handles checkpoints that might store the state_dict under 'state_dict' or 'model' keys,
    or directly as the root object.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt, .pth, etc.).

    Returns:
        The loaded state dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        TypeError: If the loaded checkpoint is not a dictionary or doesn't contain
                   a recognizable state_dict structure.
        KeyError: If common keys like 'state_dict' or 'model' are expected but not found
                  in a structured checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load onto CPU to avoid potential GPU memory issues
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            print("Extracted 'state_dict' from checkpoint.")
            return checkpoint['state_dict']
        elif 'model' in checkpoint:
            print("Extracted 'model' from checkpoint.")
            return checkpoint['model']
        else:
            # Assume the dictionary itself is the state_dict
            print(
                "Using the entire loaded dictionary as state_dict (no 'state_dict' or 'model' key found).")
            # Basic check: ensure values are tensors
            if not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                raise TypeError(
                    "Loaded dictionary contains non-Tensor values, cannot interpret as state_dict.")
            return checkpoint
    elif isinstance(checkpoint, Dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        # Handles cases where torch.load might return OrderedDict directly
        print("Loaded object is a dict of Tensors, assuming it's the state_dict.")
        # type: ignore[return-value] # Ignore type checker complaint here
        return checkpoint
    else:
        raise TypeError(
            f"Error: Loaded checkpoint is not a dictionary or does not contain a recognizable "
            f"state_dict structure. Found type: {type(checkpoint)}"
        )


def main(args: argparse.Namespace) -> None:
    """
    Main execution function. Loads, converts, and saves the checkpoint.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        # --- 1. Load Checkpoint ---
        lightning_sd = load_checkpoint_state_dict(args.input_checkpoint)

        # --- 2. Convert Keys ---
        prefix = args.prefix if args.prefix else None  # Use None if empty string
        print(
            f"Attempting key conversion. Prefix to remove: '{prefix or 'None'}'")
        detectron2_sd = convert_keys(lightning_sd, prefix)

        if not detectron2_sd:
            print("Warning: The converted state dictionary is empty. "
                  "Please check the conversion logic, skipped keys, and prefix setting.")
            # Optionally exit if an empty dict is considered an error
            # sys.exit(1)

        # --- 3. Prepare Output ---
        output_path = args.output_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            print(f"Ensuring output directory exists: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Check output file extension
        if not output_path.lower().endswith((".pth", ".pt")):
            print(f"Warning: Output path '{output_path}' does not end with '.pth' or '.pt'. "
                  "Saving with torch.save, but '.pth' is standard for Detectron2 Checkpointer.")

        # --- 4. Save Converted Checkpoint ---
        print(
            f"Saving converted state_dict ({len(detectron2_sd)} keys) to: {output_path}")
        save_obj = {"model": detectron2_sd,
                    "__author__": "convert_timm_to_detectron2",
                    "matching_heuristics": True}

        torch.save(save_obj, output_path)  # Save in a dict for compatibility

        print(
            f"\nSuccessfully converted checkpoint and saved to {output_path}")

        print("Converted keys:")
        for key in detectron2_sd.keys():
            print(f"- {key}")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except (TypeError, ValueError, KeyError) as e:
        print(f"\nError processing checkpoint: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a timm/PyTorch Lightning ResNet checkpoint state_dict to Detectron2 format.",
        # Show default values in help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_checkpoint",
        required=True,
        type=str,
        help="Path to the input timm/PyTorch Lightning checkpoint file (.ckpt, .pth, .pt)."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save the converted Detectron2 state_dict (e.g., model_final.pth)."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",  # Keep default as empty string for user convenience
        help="Prefix to remove from state_dict keys (e.g., 'model.backbone.'). "
        "Leave empty if no prefix removal is needed."
    )

    cli_args = parser.parse_args()
    main(cli_args)
