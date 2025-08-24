import argparse
import torch
import os
import sys
from typing import Dict, Any, List, Optional

# Define constants for commonly skipped key patterns in ViT
# Adjust these based on the specific timm ViT model if necessary
# Classification head and associated norm
VIT_SKIPPED_PREFIXES = ('head.', 'fc_norm.')
VIT_SKIPPED_SUFFIXES = ('.num_batches_tracked',)


def convert_vit_keys(
    timm_sd: Dict[str, torch.Tensor],
    prefix_to_remove: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Converts state_dict keys from a timm Vision Transformer (ViT) format
    to a format potentially compatible with Detectron2 ViT backbones.

    Handles prefix removal, skips auxiliary keys (like `num_batches_tracked`),
    and skips the final classification head layers. Assumes Detectron2 ViT
    backbone keys largely match timm keys after prefix removal and head skipping.

    Args:
        timm_sd: The state dictionary loaded from the timm ViT checkpoint.
        prefix_to_remove: Optional prefix string to remove from keys (e.g., "model.backbone.").
                          If None or empty, no prefix removal is attempted.

    Returns:
        A new state dictionary potentially compatible with Detectron2 ViT backbones.

    Raises:
        ValueError: If `prefix_to_remove` is specified but not found on any key.
    """
    new_sd: Dict[str, torch.Tensor] = {}
    skipped_keys: List[str] = []
    # Assume found if no prefix needed
    prefix_found_on_any_key = False if prefix_to_remove else True
    print(timm_sd.keys())
    print(f"Processing {len(timm_sd)} keys for ViT conversion...")

    for original_key, tensor_val in timm_sd.items():
        key_after_prefix = original_key

        # --- 1. Handle Prefix ---
        if prefix_to_remove:
            if original_key.startswith(prefix_to_remove):
                key_after_prefix = original_key[len(prefix_to_remove):]
                prefix_found_on_any_key = True
            else:
                # Skip keys that don't have the specified prefix
                skipped_keys.append(original_key)
                continue

        # --- 2. Skip Auxiliary and Head Keys ---
        if key_after_prefix.endswith(VIT_SKIPPED_SUFFIXES):
            skipped_keys.append(original_key)
            continue
        if key_after_prefix.startswith(VIT_SKIPPED_PREFIXES):
            skipped_keys.append(original_key)
            continue

        # --- 3. Key Mapping (Often identity for ViT after prefix/head removal) ---
        # For standard ViTs, the remaining keys (cls_token, pos_embed, patch_embed.*,
        # blocks.*, norm.*) often match Detectron2's expected names directly.
        # If specific renaming is needed for a particular D2 ViT implementation,
        # add mapping rules here similar to the ResNet converter.
        mapped_key = key_after_prefix

        if mapped_key.startswith("vit."):
            mapped_key = mapped_key[len("vit."):]

        # Finally put the Detectron2 prefix in front
        if not mapped_key.startswith("backbone.net."):
            mapped_key = "backbone.net." + mapped_key

        # Example of a potential mapping if needed (uncomment and adapt if necessary):
        # if mapped_key.startswith('some_timm_specific_name.'):
        #     mapped_key = mapped_key.replace('some_timm_specific_name.', 'd2_expected_name.', 1)

        new_sd[mapped_key] = tensor_val.cpu()  # Ensure tensors are on CPU

    # --- 4. Final Prefix Check ---
    if prefix_to_remove and not prefix_found_on_any_key:
        error_msg = (
            f"\nError: Specified prefix '{prefix_to_remove}' was not found on ANY keys "
            f"in the state dictionary.\nPlease check the --prefix argument or inspect "
            f"the checkpoint keys.\nFirst few keys found: {list(timm_sd.keys())[:5]}"
        )
        raise ValueError(error_msg)

    # --- 5. Report Skipped Keys ---
    if skipped_keys:
        print(
            f"\nSkipped {len(skipped_keys)} keys (e.g., head, auxiliary, prefix mismatch):")
        limit = 20
        for i, sk in enumerate(skipped_keys):
            print(f"- {sk}")
            if i >= limit - 1 and len(skipped_keys) > limit:
                print(f"... and {len(skipped_keys) - limit} more.")
                break

    print(f"\nSuccessfully converted {len(new_sd)} keys.")
    if not new_sd:
        print("Warning: The converted state dictionary is empty. Check input checkpoint and prefix.")
    return new_sd


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads the state dictionary from a checkpoint file.

    Handles checkpoints that might store the state_dict under 'state_dict' or 'model' keys,
    or directly as the root object.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt, .pth, .pt).

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
    # Check if it's a dict-like object containing tensors (e.g. OrderedDict)
    elif hasattr(checkpoint, 'keys') and hasattr(checkpoint, 'values') and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        print("Loaded object is a dict-like collection of Tensors, assuming it's the state_dict.")
        return checkpoint  # type: ignore[return-value]
    else:
        raise TypeError(
            f"Error: Loaded checkpoint is not a dictionary or does not contain a recognizable "
            f"state_dict structure. Found type: {type(checkpoint)}"
        )


def main(args: argparse.Namespace) -> None:
    """
    Main execution function. Loads, converts, and saves the ViT checkpoint.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        # --- 1. Load Checkpoint ---
        timm_sd = load_checkpoint_state_dict(args.input_checkpoint)

        # --- 2. Convert Keys ---
        prefix = args.prefix if args.prefix else None  # Use None if empty string
        print(
            f"Attempting ViT key conversion. Prefix to remove: '{prefix or 'None'}'")
        detectron2_sd = convert_vit_keys(timm_sd, prefix)

        if not detectron2_sd and not args.allow_empty:
            print("Error: The converted state dictionary is empty. Check input, prefix, and key names.", file=sys.stderr)
            print("If an empty output is expected (e.g., only prefix removal needed and no keys matched), use --allow-empty.", file=sys.stderr)
            sys.exit(1)
        elif not detectron2_sd and args.allow_empty:
            print(
                "Warning: The converted state dictionary is empty, but proceeding due to --allow-empty.")

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
        # Detectron2 often expects the weights under a 'model' key
        save_obj = {'model': detectron2_sd,
                    '__author__': 'convert_vit_to_detectron2'}
        torch.save(save_obj, output_path)
        print(detectron2_sd.keys())

        print(
            f"\nSuccessfully converted ViT checkpoint and saved to {output_path}")

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
        description="Convert a timm Vision Transformer (ViT) checkpoint state_dict to a Detectron2-compatible format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_checkpoint",
        required=True,
        type=str,
        help="Path to the input timm ViT checkpoint file (.ckpt, .pth, .pt)."
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
        default="",
        help="Prefix to remove from state_dict keys (e.g., 'model.backbone.', 'backbone.'). "
             "Leave empty if no prefix removal is needed."
    )
    parser.add_argument(
        "--allow-empty",
        action='store_true',
        help="Allow saving an empty state dictionary if no keys remain after conversion. "
             "Useful if the script is only used for prefix removal and might result in no matching keys."
    )

    cli_args = parser.parse_args()
    main(cli_args)
