"""
    File name: helper.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/16/2019
    Python Version: 3.7

"""

from collections import Counter
import numpy as np


def _calc_gini(pos_count, neg_count):
    total = pos_count + neg_count
    if total == 0:
        return 0
    gini = 1 - (pos_count / total) ** 2 - (neg_count / total) ** 2
    return gini


def _get_split_prob(pos_count, neg_count, total_count):
    prob = (pos_count + neg_count) / total_count
    return prob


def _predict(tree, X):
    """Generates predictions for class labels on training, validation, or test sets.

    Args:
        tree (DecisionTree): learned decision tree.
        X (ndarray): examples to generate predictions on.

    Returns:
        predictions (list): list of generated predictions.
    """
    predictions = []
    # Iterate through new observations and generate predictions
    for i in range(np.size(X, axis=0)):
        observation = X[i, :]
        node = tree
        while node.left is not None:
            if observation[node.feature_index] == 1:
                node = node.left
            else:
                node = node.right
        predictions.append(node.predicted_class)
    return predictions


def _accuracy(predictions, labels):
    """Calculate accuracy of decision tree.

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

# def get_split_benefit(feature, labels):
#     # Get total counts for positive and negative classes in dataset
#     pos_count = sum(labels)
#     neg_count = len(labels) - pos_count
#     total_count = len(labels)
#
#     # Calculate initial uncertainty
#     initial_u = get_gini(pos_count, neg_count)
#
#     # Create list of tuples with class label and feature value for each observation
#     # i.e. (1,1) represents class label == 1 and feature label == 1
#     zipped = list(zip(labels, feature))
#
#     # Create dictionary with counts of tuples for each possible class label and feature
#     # value pair
#     label_value_count = dict(Counter(element for element in zipped))
#
#     # Get counts for positive and negative class in each branch after split
#     # Left count in positive class ('class' == 1)
#     if (1, 1) in label_value_count:
#         left_pos = label_value_count[(1, 1)]
#     else:
#         left_pos = 0
#     # Left count in negative class ('class' == 0)
#     if (0, 1) in label_value_count:
#         left_neg = label_value_count[(0, 1)]
#     else:
#         left_neg = 0
#     # Right count in positive class ('class' == 1)
#     if (1, 0) in label_value_count:
#         right_pos = label_value_count[(1, 0)]
#     else:
#         right_pos = 0
#
#     if (0, 0) in label_value_count:
#         right_neg = label_value_count[(0, 0)]
#     else:
#         right_neg = 0
#
#     # calculate probabilities for left and right children
#     p_left = get_split_prob(left_pos, left_neg, total_count)
#     p_right = get_split_prob(right_pos, right_neg, total_count)
#
#     # get left and right uncertainties
#     left_u = get_gini(left_pos, left_neg)
#     right_u = get_gini(right_pos, left_pos)
#
#     # calculate benefit of split
#     benefit = initial_u - p_left * left_u - p_right * right_u
#     return benefit
#
#
# # To get root node
# def get_best_split(train):
#     # Get list of feature names and list of class labels
#     feature_names = train.columns.to_list()
#     labels = train['class'].to_list()
#     # Create dictionary to store features and uncertainties
#     feature_uncertainty = {k: 0 for k in feature_names}
#
#     for feature_name in feature_names:
#         feature = train[feature_name].to_list()
#         feature_uncertainty[feature_name] = get_split_benefit(feature, labels)
#     # Extract feature that provides best benefit when split on
#     best_feature = max(feature_uncertainty, key=feature_uncertainty.get)
#     return best_feature
