"""
K-nearest neighbors evaluation for quantitative assessment of learned RF signal representations.

This module provides a standardized k-NN classifier evaluation framework specifically designed for assessing the quality of features learned through self-supervised pretraining on RF signals. It serves as a crucial component in the evaluation pipeline by offering a simple yet effective method to measure how well learned representations capture semantic relationships between different signal types or classes. The k-NN evaluation is particularly valuable in SSL contexts because it provides a parameter-free assessment of representation quality without requiring additional model training or fine-tuning. Typical use cases include comparing different SSL methods, validating pretraining effectiveness before downstream fine-tuning, and selecting optimal feature extraction layers from pretrained models. The module integrates with the broader evaluation ecosystem by consuming feature vectors from various SSL models and providing quantitative metrics that can be used for model selection, hyperparameter optimization, and research comparisons across different RF signal processing approaches.
"""

from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

__all__ = ["EvaluateKNN"]


class EvaluateKNN:
    """
    Evaluates feature representation quality using k-nearest neighbors classification.

    Provides a standardized evaluation framework for assessing learned representations through classification accuracy on held-out test data.

    :param x: Feature vectors extracted from pretrained models for evaluation
    :type x: List[np.ndarray]
    :param y: Ground truth class labels corresponding to each feature vector
    :type y: List[str]
    :param split: Fraction of data to use for training the k-NN classifier
    :type split: float
    :param shuffel: Whether to randomly shuffle the dataset before train/test splitting
    :type shuffel: bool
    :param n_neighbors: Number of nearest neighbors to consider in k-NN classification
    :type n_neighbors: int
    :param verbose: Whether to print detailed evaluation progress and statistics
    :type verbose: bool
    """

    def __init__(self,
                 x: List[np.ndarray],
                 y: List[str],
                 split: float = 0.8,
                 shuffel: bool = True,
                 n_neighbors: int = 50,
                 verbose: bool = True):
        """
        Initializes k-NN evaluation with feature vectors and corresponding labels.

        Automatically splits data into train/test sets and prepares the evaluation framework with specified hyperparameters.

        :param x: Feature vectors extracted from pretrained models for evaluation
        :type x: List[np.ndarray]
        :param y: Ground truth class labels corresponding to each feature vector
        :type y: List[str]
        :param split: Fraction of data to use for training the k-NN classifier
        :type split: float
        :param shuffel: Whether to randomly shuffle the dataset before train/test splitting
        :type shuffel: bool
        :param n_neighbors: Number of nearest neighbors to consider in k-NN classification
        :type n_neighbors: int
        :param verbose: Whether to print detailed evaluation progress and statistics
        :type verbose: bool
        """
        self.verbose = verbose

        if self.verbose:
            print(f"Evaluating with KNN (k={n_neighbors})")
            print(f"Total samples: {len(x)}")
            # Count unique classes
            unique_classes = set(y)
            print(f"Number of classes: {len(unique_classes)}")

        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, train_size=split, shuffle=shuffel, random_state=42)

        if self.verbose:
            print(f"Train set: {len(self.x_train)} samples")
            print(f"Test set: {len(self.x_test)} samples")

        self.n_neighbors = n_neighbors

    def evaluate(self) -> float:
        """
        Performs k-NN classification evaluation and returns test accuracy.

        Trains a k-NN classifier on the training split and evaluates performance on the held-out test set.

        :returns: Classification accuracy score on the test set
        :rtype: float
        """
        if self.verbose:
            print("Fitting KNN classifier...")

        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Fit the model
        knn.fit(self.x_train, self.y_train)

        if self.verbose:
            print("Predicting on test set...")

        # Predict the labels
        pred_labels = knn.predict(self.x_test)

        # Calculate accuracy
        acc = accuracy_score(self.y_test, pred_labels)

        if self.verbose:
            print(f"KNN Accuracy: {acc:.4f}")

        return acc
