"""
    File name: functions.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/17/2019
    Python Version: 3.7

"""

import random

import numpy as np


def _gini(y, n_classes):
    """Calculate gini impurity for a node.

    Args:
        y (ndarray): labels for data at node.
        n_classes (int): number of classes.
        boosted (bool): indicator for using boosted trees.
        class_weights (list): if using boosted trees, list of total weight in each class at node.

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
    # Store list of predictions for each tree in list.
    predictions = []
    for tree in trees:
        predictions.append(_predict(tree, X))
    # Zip together predictions for each instance from each tree to take majority vote.
    zipped = list(zip(*predictions))
    # Get tuple with number of votes for each class, i.e. (3, 2) means 3 votes for 0 and 2 votes for 1.
    class_votes = [(item.count(0), item.count(1)) for item in zipped]
    # Generate list with predictions based on majority vote, if tie randomly generate value in {0, 1}.
    rf_predictions = [int(np.argmax(item)) if item[0] != item[1] else random.randint(0, 1) for item in class_votes]
    return rf_predictions


def _predict_adaboost(clfs, clf_alphas, X):
    """Generates predictions from boosted trees (AdaBoost) for class labels on training, validation, or test sets.

    Args:
        clfs (list): list of trained base classifiers.
        clf_alphas (list): list of alphas from trained base classifiers.
        X (ndarray): examples to generate predictions on.

    Returns:
        predictions (list): list of predicted values.
    """
    N = np.size(X, axis=0)
    y = np.zeros(N)
    for (clf, alpha) in zip(clfs, clf_alphas):
        preds = _predict(clf, X)
        preds = np.array([-1 if y == 0 else 1 for y in preds])
        y = y + alpha * preds
    predictions = list(np.sign(y))
    # Convert class labels from {-1, 1} back to {0, 1}.
    predictions = [0 if y == -1 else 1 for y in predictions]
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
