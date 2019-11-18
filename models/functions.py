"""
    File name: functions.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/17/2019
    Python Version: 3.7

"""

import numpy as np


def _gini(y, n_classes):
    """Calculate gini impurity for a node.

    Args:
        y (ndarray): labels for data at node.
        n_classes (int): number of classes.

    Returns:
        gini_index (float): computed gini index.
    """
    m = y.size
    gini_index = 1 - sum((np.sum(y == i) / m) ** 2 for i in range(n_classes))
    return gini_index


def _predict(tree, X):
    """Generates predictions for class labels on training, validation, or test sets.

    Args:
        tree (DecisionTree): learned decision tree.
        X (ndarray): examples to generate predictions on.

    Returns:
        predictions (list): list of generated predictions.
    """
    predictions = []
    # for i in range(np.size(X, axis=0)):
    for observation in X:
        # observation = X[i, :]
        node = tree
        while node.left is not None:
            if observation[node.feature_index] == 1:
                node = node.left
            else:
                node = node.right
        predictions.append(node.predicted_class)
    return predictions


def _accuracy(predictions, labels):
    """Calculate accuracy of decision tree predictions.

    Args:
        predictions (list or ndarray): list or ndarray of class predictions in {0, 1}
        labels (list or ndarray): list or ndarray of true class labels.

    Returns:
        accuracy (float): calculated accuracy.
    """
    # Ensure lists are coerced to ndarrays of integers.
    predictions = np.array(predictions, dtype=int)
    labels = np.array(labels, dtype=int)
    correct = (labels == predictions)
    acc = correct.sum() / np.size(correct)
    return acc
