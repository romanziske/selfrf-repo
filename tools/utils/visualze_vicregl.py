import matplotlib.pyplot as plt
import torch
import numpy as np
from selfrf.pretraining.config import parse_training_config
from selfrf.pretraining.factories import build_dataloader


def visualize_vicregl_views(config, num_samples=3):
    """
    Visualizes VICRegL views (global and local crops) from the dataloader.

    :param config: Training configuration
    :param num_samples: Number of samples to visualize
    """
    # Build dataloader
    datamodule = build_dataloader(config)
    datamodule.setup()

    # Get a batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    (views, grids), targets = batch

    print(f"Number of views: {len(views)}")
    print(f"View shapes: {[view.shape for view in views]}")
    print(f"Grid shapes: {[grid.shape for grid in grids]}")

    # Visualize first few samples
    for sample_idx in range(min(num_samples, len(targets))):
        visualize_sample_views(views, grids, targets, sample_idx)


def visualize_sample_views(views, grids, targets, sample_idx):
    """
    Visualizes all views for a single sample.

    :param views: List of view tensors [batch_size, channels, height, width]
    :param grids: List of grid tensors  
    :param targets: Target labels
    :param sample_idx: Which sample in the batch to visualize
    """
    num_views = len(views)

    # Assuming 2 global + 6 local views
    n_global = 2
    n_local = num_views - n_global

    # Create subplot grid
    fig, axes = plt.subplots(2, max(n_global, n_local), figsize=(15, 8))
    fig.suptitle(
        f'Sample {sample_idx}, Target: {targets[sample_idx].item()}', fontsize=16)

    # Plot global views
    for i in range(n_global):
        view_tensor = views[i][sample_idx]  # [channels, height, width]

        # Convert to numpy and handle channels
        if view_tensor.shape[0] == 1:
            # Single channel - grayscale
            img = view_tensor.squeeze(0).cpu().numpy()
            cmap = 'viridis'
        else:
            # Multiple channels - take first or convert to RGB
            img = view_tensor[0].cpu().numpy()
            cmap = 'viridis'

        axes[0, i].imshow(img, cmap=cmap)
        axes[0, i].set_title(f'Global View {i+1}\n{view_tensor.shape[1:]}')
        axes[0, i].axis('off')

        # Overlay grid if available
        # if i < len(grids):
        #    overlay_grid(axes[0, i], grids[i][sample_idx], img.shape)

    # Hide unused global view plots
    for i in range(n_global, max(n_global, n_local)):
        axes[0, i].axis('off')

    # Plot local views
    for i in range(n_local):
        view_idx = n_global + i
        view_tensor = views[view_idx][sample_idx]

        # Convert to numpy
        if view_tensor.shape[0] == 1:
            img = view_tensor.squeeze(0).cpu().numpy()
            cmap = 'viridis'
        else:
            img = view_tensor[0].cpu().numpy()
            cmap = 'viridis'

        col_idx = i if i < max(
            n_global, n_local) else i % max(n_global, n_local)
        axes[1, col_idx].imshow(img, cmap=cmap)
        axes[1, col_idx].set_title(
            f'Local View {i+1}\n{view_tensor.shape[1:]}')
        axes[1, col_idx].axis('off')

        # Overlay grid if available
       # if view_idx < len(grids):
       #     overlay_grid(axes[1, col_idx], grids[view_idx]
       #                  [sample_idx], img.shape)

    # Hide unused local view plots
    for i in range(n_local, max(n_global, n_local)):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def overlay_grid(ax, grid_tensor, img_shape):
    """
    Overlays spatial grid points on the image with proper coordinate mapping.

    :param ax: Matplotlib axis
    :param grid_tensor: Grid coordinates [grid_h, grid_w, 2]
    :param img_shape: Shape of the image (height, width)
    """
    if grid_tensor is None:
        print("Grid tensor is None")
        return

    grid_np = grid_tensor.cpu().numpy()

    print(f"\n=== Grid Debug ===")
    print(f"Grid shape: {grid_np.shape}")
    print(f"Image shape: {img_shape}")
    print(f"Grid coordinates sample:")
    print(f"  Top-left: ({grid_np[0,0,0]:.1f}, {grid_np[0,0,1]:.1f})")
    print(f"  Top-right: ({grid_np[0,-1,0]:.1f}, {grid_np[0,-1,1]:.1f})")
    print(f"  Bottom-left: ({grid_np[-1,0,0]:.1f}, {grid_np[-1,0,1]:.1f})")
    print(f"  Bottom-right: ({grid_np[-1,-1,0]:.1f}, {grid_np[-1,-1,1]:.1f})")

    # Extract x and y coordinates
    x_coords = grid_np[:, :, 0].flatten()
    y_coords = grid_np[:, :, 1].flatten()

    print(f"Original X range: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
    print(f"Original Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}]")
    print(
        f"Image dimensions: {img_shape[1]} x {img_shape[0]} (width x height)")

    # **FIX: Map grid coordinates to the cropped image space**
    if x_coords.max() > img_shape[1] or y_coords.max() > img_shape[0]:
        print("🔧 Grid coordinates exceed image bounds - mapping to image space")

        # Find the bounding box of the grid
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Map to [0, image_size] coordinate system
        x_coords_mapped = (x_coords - x_min) / (x_max - x_min) * img_shape[1]
        y_coords_mapped = (y_coords - y_min) / (y_max - y_min) * img_shape[0]

        print(
            f"Mapped X range: [{x_coords_mapped.min():.1f}, {x_coords_mapped.max():.1f}]")
        print(
            f"Mapped Y range: [{y_coords_mapped.min():.1f}, {y_coords_mapped.max():.1f}]")

        x_coords = x_coords_mapped
        y_coords = y_coords_mapped

    elif x_coords.max() <= 1.0 and y_coords.max() <= 1.0:
        print(
            "📝 Coordinates appear to be normalized [0,1] - scaling to pixels")
        # Convert to pixel coordinates
        x_coords = x_coords * img_shape[1]  # width
        y_coords = y_coords * img_shape[0]  # height
        print(
            f"After scaling: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    else:
        print("📝 Coordinates appear to be in correct pixel space")

    # Plot grid points with better visibility
    ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8, marker='o',
               edgecolors='white', linewidths=2)

    # Draw grid lines for structure
    grid_h, grid_w = grid_np.shape[:2]

    # Reshape for line drawing
    grid_x = x_coords.reshape(grid_h, grid_w)
    grid_y = y_coords.reshape(grid_h, grid_w)

    # Draw horizontal lines
    for i in range(grid_h):
        ax.plot(grid_x[i, :], grid_y[i, :], 'r-', alpha=0.6, linewidth=2)

    # Draw vertical lines
    for j in range(grid_w):
        ax.plot(grid_x[:, j], grid_y[:, j], 'r-', alpha=0.6, linewidth=2)

    print("✅ Grid overlay complete")


def overlay_grid_simple(ax, grid_tensor, img_shape):
    """
    Simple grid overlay that just normalizes coordinates to image space.
    """
    if grid_tensor is None:
        return

    grid_np = grid_tensor.cpu().numpy()

    # Extract coordinates
    x_coords = grid_np[:, :, 0].flatten()
    y_coords = grid_np[:, :, 1].flatten()

    # Always normalize to [0, img_size] range
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Avoid division by zero
    if x_max > x_min:
        x_normalized = (x_coords - x_min) / (x_max - x_min) * img_shape[1]
    else:
        x_normalized = np.full_like(x_coords, img_shape[1] / 2)

    if y_max > y_min:
        y_normalized = (y_coords - y_min) / (y_max - y_min) * img_shape[0]
    else:
        y_normalized = np.full_like(y_coords, img_shape[0] / 2)

    # Plot grid
    ax.scatter(x_normalized, y_normalized, c='red', s=30, alpha=0.8, marker='o',
               edgecolors='white', linewidths=1)

    # Draw grid lines
    grid_h, grid_w = grid_np.shape[:2]
    grid_x = x_normalized.reshape(grid_h, grid_w)
    grid_y = y_normalized.reshape(grid_h, grid_w)

    for i in range(grid_h):
        ax.plot(grid_x[i, :], grid_y[i, :], 'r-', alpha=0.5, linewidth=1)
    for j in range(grid_w):
        ax.plot(grid_x[:, j], grid_y[:, j], 'r-', alpha=0.5, linewidth=1)
