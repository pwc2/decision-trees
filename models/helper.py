"""
    File name: helper.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/16/2019
    Python Version: 3.7

"""

from collections import Counter


def calc_gini(pos_count, neg_count):
    total = pos_count + neg_count
    gini = 1 - (pos_count / total) ** 2 - (neg_count / total) ** 2
    return gini


def get_split_prob(pos_count, neg_count, total_count):
    prob = (pos_count + neg_count) / total_count
    return prob


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
