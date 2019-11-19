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
    for observation in X:
        node = tree
        while node.left is not None:
            if observation[node.feature_index] == 1:
                node = node.left
            else:
                node = node.right
        predictions.append(node.predicted_class)
    return predictions


def _predict_rf(trees, X):
    """Generates predictions from RandomForest for class labels on training, validation, or test sets using majority
    vote from ensemble of trees.

    Args:
        trees (list of DecisionTree): list of learned decision trees from random forest.
        X (ndarray): examples to generate predictions on.

    Returns:
        rf_predictions (list): list of generated predictions from majority vote.
    """
    # Store predictions for each tree.
    predictions = []
    for tree in trees:
        tree_predictions = []
        for observation in X:
            node = tree
            while node.left is not None:
                if observation[node.feature_index] == 1:
                    node = node.left
                else:
                    node = node.right
            tree_predictions.append(node.predicted_class)
        predictions.append(tree_predictions)

    # Zip together predictions for each instance from each tree to take majority vote.
    zipped = list(zip(*predictions))
    # Get tuple with number of votes for each class, i.e. (3, 2) means 3 votes for 0 and 2 votes for 1.
    class_votes = [(item.count(0), item.count(1)) for item in zipped]
    # Generate list with predictions based on majority vote.
    rf_predictions = [int(np.argmax(item)) for item in class_votes]
    return rf_predictions


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
