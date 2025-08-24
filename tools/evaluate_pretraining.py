from typing import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

from selfrf.pretraining.evaluation import EvaluateKNN, VisualizeEmbeddings
from selfrf.pretraining.config import EvaluationConfig, parse_evaluation_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_backbone
from selfrf.pretraining.utils.utils import get_class_list


def convert_idx_to_name(idx: int, config: EvaluationConfig) -> str:
    """Convert index to either class or family name"""
    return get_class_list(config)[idx]


def load_model_weights(model: torch.nn.Module, checkpoint: dict | OrderedDict, config: EvaluationConfig):
    """Loads weights from a checkpoint into the model, handling potential state_dict formats and prefixes."""
    # Handle state_dict format if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint  # Assume the checkpoint is just the state_dict
    else:
        raise TypeError(
            f"Checkpoint format not recognized: {type(checkpoint)}")

    # --- Add state dict processing for both ViT and ResNet ---
    new_state_dict = OrderedDict()

    # Define possible prefixes for different model types
    prefixes_to_remove = [
        'backbone.vit.',      # Lightning MAE ViT model
        'vit.',               # Direct ViT backbone
        'backbone.resnet.',   # Lightning ResNet model
        'backbone.',          # Generic backbone prefix
        'resnet.',            # Direct ResNet backbone
    ]

    found_prefix = False
    for k, v in state_dict.items():
        # Try each prefix
        for prefix in prefixes_to_remove:
            if k.startswith(prefix):
                name = k[len(prefix):]
                new_state_dict[name] = v
                found_prefix = True
                break

        # If no prefix matched, keep the original key
        if not any(k.startswith(prefix) for prefix in prefixes_to_remove):
            new_state_dict[k] = v

    if not found_prefix:
        print(
            "Warning: Could not find expected prefix in checkpoint keys. Using keys as-is.")
        # Show first 5 keys
        print(f"Available keys: {list(state_dict.keys())[:5]}...")
    else:
        print(f"Successfully processed checkpoint with prefix removal.")

    # Load the processed weights
    try:
        # Use strict=False because classification heads might be missing
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("State dict loading results:", msg)

        if not msg.missing_keys and not msg.unexpected_keys:
            print("Weights loaded successfully.")
        elif msg.missing_keys:
            # Expected missing keys for different model types
            expected_missing = {
                'head.weight', 'head.bias',           # ViT classification head
                'fc.weight', 'fc.bias',               # ResNet classification head
            }
            other_missing = [
                k for k in msg.missing_keys if k not in expected_missing]

            if not other_missing:
                print(
                    f"Weights loaded successfully. Missing keys (expected): {msg.missing_keys}")
            else:
                print(f"Warning: Unexpected missing keys: {other_missing}")
                print(f"Expected missing keys: {expected_missing}")

        if msg.unexpected_keys:
            print(
                f"Warning: Unexpected keys in checkpoint: {msg.unexpected_keys}")

    except Exception as e:
        print(f"Error loading state dict: {e}")
        print(
            f"Model expects keys like: {list(model.state_dict().keys())[:5]}...")
        print(
            f"Checkpoint has keys like: {list(new_state_dict.keys())[:5]}...")
        raise e


def evaluate(config: EvaluationConfig):

    if not config.model_path:
        raise ValueError("model_path is required for evaluation")

    datamodule = build_dataloader(config)

    model = build_backbone(config)

    if config.model_path.lower() == "random" or not config.model_path:
        print("Using randomly initialized weights")
        # Model already has random weights from initialization
    else:
        print(f"Loading weights from {config.model_path}")
        checkpoint = torch.load(
            config.model_path,
            map_location=config.device,
            weights_only=False,  # Set to False if it might contain code, True if only weights
        )
        load_model_weights(model, checkpoint, config)

    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()  # Remove classification head if it exists
    model = model.to(config.device)
    model.eval()

    representations = []
    labels = []
    datamodule.prepare_data()
    datamodule.setup("fit")
    val_dataloader = datamodule.val_dataloader()
    with torch.no_grad():  # No gradient needed

        for x, targets in tqdm(val_dataloader):

            x = x.to(config.device)

            z = model(x)  # Run inference
            # Move features back to CPU and convert to numpy
            representations.extend(z.cpu().numpy())

            # Use .item() to get Python number from tensor
            labels.extend(
                [convert_idx_to_name(t.item(), config) for t in targets])

    representations = np.array(representations)
    labels = np.array(labels)
    print(
        f"Finished calculating representations (shape {representations.shape})")

    print("Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"plot_{model_name}.png"
    visualizer = VisualizeEmbeddings(
        x=representations,
        y=labels,
        class_list=get_class_list(config),
    )

    visualizer.visualize(
        method="tsne",
        save_path="tsne_" + plot_path,
        perplexity=50
    )
    print(f"t-SNE plot saved.")

    visualizer.visualize(
        method="umap",
        save_path="umap_" + plot_path,
        n_neighbors=8,
        min_dist=0.5,
        metric='cosine',
    )
    print(f"UMAP plot saved.")

    print("Start KNN evaluation...")
    EvaluateKNN(representations, labels,
                n_neighbors=config.n_neighbors).evaluate()


if __name__ == "__main__":
    config = parse_evaluation_config()
    print_config(config)
    evaluate(config)
