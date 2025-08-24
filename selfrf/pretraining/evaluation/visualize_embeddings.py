"""
Dimensionality reduction visualization tools for qualitative analysis of learned RF signal representations.

This module provides comprehensive visualization capabilities for analyzing high-dimensional feature embeddings learned through self-supervised pretraining on RF signals, using state-of-the-art dimensionality reduction techniques including t-SNE, PCA, and UMAP. It serves as a crucial component for qualitative evaluation of representation learning by enabling researchers to visually inspect how well different signal classes cluster in the learned feature space. The visualization tools are particularly valuable for understanding the semantic organization of learned representations, identifying potential failure modes in SSL pretraining, and generating publication-ready figures that demonstrate the effectiveness of different pretraining strategies. Typical use cases include comparing clustering quality across different SSL methods, analyzing the impact of various augmentation strategies on representation structure, and providing intuitive visual evidence of successful representation learning for research presentations and publications. The module integrates seamlessly with the broader evaluation ecosystem by consuming feature vectors from any SSL model and producing high-quality visualizations that complement quantitative evaluation metrics from the k-NN evaluation framework.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Optional, List

# suppress sklearn's force_all_finite → ensure_all_finite warning
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
)

# suppress UMAP's "n_jobs value 1 overridden …" warning
warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value.*overridden.*",
    category=UserWarning,
)

__all__ = ["VisualizeEmbeddings"]


class VisualizeEmbeddings:
    """
    Creates 2D visualizations of high-dimensional embeddings using dimensionality reduction techniques.

    Supports multiple projection methods including t-SNE, PCA, and UMAP with automatic color mapping for different classes.

    :param x: High-dimensional feature embeddings to visualize
    :type x: np.ndarray
    :param y: Class labels for each embedding vector
    :type y: np.ndarray
    :param class_list: List of all possible class names in consistent order
    :type class_list: List[str]
    """

    def __init__(
        self,
        x: np.ndarray,         # shape (n_samples, n_features)
        y: np.ndarray,         # shape (n_samples,), class labels
        class_list: List[str],  # all possible labels in order
    ):
        """
        Initializes visualization framework with embeddings and class information.

        Automatically creates color mapping for different classes using rainbow colormap for visual distinction.

        :param x: High-dimensional feature embeddings to visualize
        :type x: np.ndarray
        :param y: Class labels for each embedding vector
        :type y: np.ndarray
        :param class_list: List of all possible class names in consistent order
        :type class_list: List[str]
        """
        self.x = x
        self.y = y
        self.class_list = class_list

        # create a color map for each label
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_list)))
        self.color_dict = dict(zip(class_list, colors))

    def visualize(
        self,
        method: str = 'tsne',      # 'tsne', 'pca' or 'umap'
        save_path: Optional[str] = None,
        **embed_kwargs,            # passed to TSNE/PCA/UMAP constructors
    ):
        """
        Projects high-dimensional embeddings to 2D space and creates scatter plot visualization.

        Uses specified dimensionality reduction method to create interpretable 2D visualization with class-based coloring.

        :param method: Dimensionality reduction technique to use
        :type method: str
        :param save_path: Optional file path to save the generated figure
        :type save_path: Optional[str]
        :param embed_kwargs: Additional keyword arguments passed to the dimensionality reduction constructor
        :returns: Matplotlib figure object containing the visualization
        :rtype: matplotlib.figure.Figure
        :raises ValueError: If method is not one of 'tsne', 'pca', or 'umap'
        """
        # choose projector
        if method == 'tsne':
            projector = TSNE(n_components=2, random_state=42, **embed_kwargs)
        elif method == 'pca':
            projector = PCA(n_components=2, random_state=42, **embed_kwargs)
        elif method == 'umap':
            projector = umap.UMAP(
                n_components=2, random_state=42, **embed_kwargs)
        else:
            raise ValueError("method must be 'tsne', 'pca' or 'umap'")

        # fit & transform
        proj2d = projector.fit_transform(self.x)

        # plot
        fig, ax = plt.subplots(figsize=(16, 9))
        for label, color in self.color_dict.items():
            mask = (self.y == label)
            ax.scatter(
                proj2d[mask, 0], proj2d[mask, 1],
                c=[color], label=label, alpha=0.6
            )

        ax.set_title(f"{method.upper()} projection of embeddings")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
