"""
    File name: node.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/17/2019
    Python Version: 3.7

"""
import numpy as np


class Node:
    """Class to construct node object for decision tree.

    """

    def __init__(self, n, class_distribution, gini_index, boost=False):
        """Constructs node object with number of observations at node, a list with the count of observations in each
        class at node, and gini impurity at node.

        Args:
            n (int): sample size of data at node.
            class_distribution (list): list with the count of observations in each class (0 and 1) at node,
            or for boosted trees a list with the sum of the weights in each class (0 and 1).
            gini_index (float): gini impurity at node.

        Returns:
            None
        """
        self.n = n
        self.class_distribution = class_distribution
        self.gini_index = gini_index
        self.predicted_class = int(np.argmax(self.class_distribution))
        self.feature_index = None
        self.left = None
        self.right = None
