"""
    File name: decision_tree.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/16/2019
    Python Version: 3.7

"""
import numpy as np
from collections import Counter
from models.helper import _calc_gini, _get_split_prob, _accuracy, _predict


class Node:
    """Class to construct node object for decision tree.

    """

    def __init__(self, n, n_classes, gini, predicted_class):
        """Constructs node object.

        Args:
            n (int): sample size of data at node.
            n_classes (list): list with the number of observations in each class (0 and 1) at node.
            gini (float): gini impurity at node.
            predicted_class (int): predicted class at node based on majority class at node.

        Returns:
            None
        """
        self.n = n
        self.n_classes = n_classes
        self.gini = gini
        self.predicted_class = predicted_class
        self.feature_index = None
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, train, validation, test, label, max_depth=None):
        """Constructs DecisionTree object.

        Args:
            train (DataFrame): DataFrame with training set observations.
            validation (DataFrame): DataFrame with validation set observations.
            test (DataFrame): DataFrame with test set observations.
            label (str): String with column name for class labels in train and validation sets.
            max_depth (int): Maximum depth to grow decision tree to.

        Returns:
            None
        """
        # Extract features and labels from datasets
        self.train_features = train.drop(label, axis=1)
        self.train_labels = train[label]
        self.validation_features = validation.drop(label, axis=1)
        self.validation_labels = validation[label]
        self.test_features = test

        self.n_classes = len(set(self.train_labels))
        self.n_features = np.size(self.train_features.to_numpy(), axis=1)
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        """Calculate gini impurity for a node.

        Args:
            y (ndarray): labels for data at node.

        Returns:
            gini (float): computed gini index.
        """
        m = y.size
        gini = 1 - sum((np.sum(y == i) / m) ** 2 for i in range(self.n_classes))
        return gini

    def train(self):
        """Learns decision tree from training data and generate predictions and accuracy on training and validation
        sets.

        Returns: results (dict): dictionary with predictions and accuracy for training and validation sets.
        """
        # self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        # self.n_features_ = X.shape[1]
        X_train = self.train_features.to_numpy()
        y_train = self.train_labels.to_numpy()
        X_val = self.validation_features.to_numpy()
        y_val = self.validation_labels.to_numpy()

        self.tree = self.grow_tree(X_train, y_train, depth=0)

        # print('Left children: ')
        # node = self.tree
        # while node.left is not None:
        #     print(node.left.feature_index)
        #     node = node.left
        #
        # print('Right children: ')
        # node = self.tree
        # while node.right is not None:
        #     print(node.right.feature_index)
        #     node = node.right

        # Calculate predictions and accuracy on train and validation sets.
        train_predictions = _predict(self.tree, X_train)
        val_predictions = _predict(self.tree, X_val)
        # train_predictions = self.predict(X_train)
        # val_predictions = self.predict(X_val)
        train_accuracy = _accuracy(train_predictions, y_train)
        val_accuracy = _accuracy(val_predictions, y_val)
        results = {'max_depth': self.max_depth,
                   'train_accuracy': train_accuracy,
                   'val_accuracy': val_accuracy,
                   'train_predictions': train_predictions,
                   'val_predictions': val_predictions, }
        return results

    def grow_tree(self, X, y, depth=0):
        """Build decision tree with recursive splitting.

        Args:
            X (ndarray): training set without class labels.
            y (ndarray) : class labels for training set.
            depth (int) : starting depth of decision tree.
        Returns:
            node (Node): root node of learned decision tree.
        """

        # Get number of training observations in current node with each class label 0 and 1.
        n_class = [np.sum(y == i) for i in range(self.n_classes)]
        # Predicted class will be the majority class.
        pred_class = int(np.argmax(n_class))
        # Instantiate node.
        node = Node(n=y.size,
                    n_classes=n_class,
                    gini=self.gini(y),
                    predicted_class=pred_class)
        # Perform recursive splitting to max depth.
        if depth < self.max_depth:
            current_gini, split_index = self.best_split(X, y)
            # Get indices for data and labels to go to the left child, send the rest to the right child
            if split_index is not None:
                index_left = X[:, split_index] == 1
                X_left, y_left = X[index_left], y[index_left]
                X_right, y_right = X[~index_left], y[~index_left]
                node.gini = current_gini
                node.feature_index = split_index
                node.left = self.grow_tree(X_left, y_left, depth=depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth=depth + 1)
        return node

    def best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the Gini impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.

        Args:
            X (ndarray): training set without class labels.
            y (ndarray) : class labels for training set.
        Returns:
            current_gini (float): gini impurity from best split.
            best_index (int): integer for column index for feature split on.
        """
        # Make sure there are at least two elements before splitting node
        m = y.size
        if m <= 1:
            return None, None

        current_gini = self.gini(y)

        # Iterate through all features and calculate gini impurity from resulting split
        split_index = None
        for index in range(self.n_features):
            feature = X[:, index]

            # # Create list of tuples with class label and feature value for each observation
            # # i.e. (1,1) represents class label == 1 and feature label == 1
            # zipped = list(zip(feature, y))
            #
            # # Go left when feature value is 1, right when it is 0
            # left = [x for x in zipped if x[0] == 1]
            # right = [x for x in zipped if x[0] == 0]
            #
            # # Create lists with counts for classes for both left and right, indices are same as class labels
            # num_left = [len([x for x in left if x[1] == 0]), len([x for x in left if x[1] == 1])]
            # num_right = [len([x for x in right if x[1] == 0]), len([x for x in right if x[1] == 1])]
            #
            # m = y.size
            # gini = 1 - sum((np.sum(y == i) / m) ** 2 for i in range(self.n_classes))

            # Get indices to go to left child if feature value is 1, otherwise right child
            left_idx = feature == 1
            left_y = y[left_idx]
            # left_x = feature[left_idx]
            right_y = y[~left_idx]
            # right_x = feature[~left_idx]

            if len(left_y) == 0:
                gini_left = 0
            else:
                gini_left = self.gini(left_y)
            if len(right_y) == 0:
                gini_right = 0
            else:
                gini_right = self.gini(right_y)

            prob_left = len(left_y) / len(y)
            prob_right = len(right_y) / len(y)

            # if gini_left == 0:
            #     prob_left = 1
            # else:
            #     prob_left = len(left_y)/len(y)
            #
            # # If node is pure, then gini impurity will be zero
            # if 0 in num_left:
            #     gini_left = 0
            #     prob_left = 1
            # else:
            #     gini_left = 1 - sum((n / sum(num_left)) ** 2 for n in num_left)
            #     prob_left = sum(num_left) / total_n
            #
            # if 0 in num_right:
            #     gini_right = 0
            #     prob_right = 1
            # else:
            #     gini_right = 1 - sum((n / sum(num_right)) ** 2 for n in num_right)
            #     prob_right = sum(num_right) / total_n

            gini = prob_left * gini_left + prob_right * gini_right

            # Update current gini value if the split is beneficial, save feature name associated with split
            if gini < current_gini:
                current_gini = gini
                split_index = index
        return current_gini, split_index

    # def predict(self, X):
    #     """Predict class for X."""
    #     return [self._predict(inputs) for inputs in X]
    #
    # def _predict(self, inputs):
    #     """Predict class for a single sample."""
    #     node = self.tree
    #     while node.left:
    #         if inputs[node.feature_index] == 1:
    #             node = node.left
    #         else:
    #             node = node.right
    #     return node.predicted_class
