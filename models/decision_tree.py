"""
    File name: decision_tree.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/19/2019
    Python Version: 3.7

"""
import numpy as np

from models.functions import _gini, _predict, _accuracy
from models.node import Node


class DecisionTree:
    """Class to construct decision tree object.

    """

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
        # Extract features and labels from data sets
        self.train_set = train
        self.train_features = train.drop(label, axis=1)
        self.train_labels = train[label]
        self.validation_features = validation.drop(label, axis=1)
        self.validation_labels = validation[label]
        self.test_set = test

        self.n_classes = len(set(self.train_labels))
        self.n_features = np.size(self.train_features.to_numpy(), axis=1)
        self.max_depth = max_depth
        self.tree = None

    def predict(self, X):
        """Generate predictions on training, validation or test set.

        Args:
            X (ndarray): set to generated predictions on.

        Returns:
            predictions (list): list of generated predictions.
        """
        predictions = _predict(self.tree, X)
        return predictions

    def train(self):
        """Learns decision tree from training data and generate predictions and accuracy on training and validation
        sets.

        Returns:
            results (dict): dictionary with predictions and accuracy for training and validation sets.
        """
        X_train = self.train_features.to_numpy()
        y_train = self.train_labels.to_numpy()
        X_val = self.validation_features.to_numpy()
        y_val = self.validation_labels.to_numpy()

        # Learn decision tree.
        self.tree = self.fit_tree(X_train, y_train, depth=0)

        # Calculate predictions and accuracy on train and validation sets.
        train_predictions = self.predict(X_train)
        val_predictions = self.predict(X_val)
        train_accuracy = _accuracy(train_predictions, y_train)
        val_accuracy = _accuracy(val_predictions, y_val)

        results = {'max_depth': self.max_depth,
                   'train_accuracy': train_accuracy,
                   'val_accuracy': val_accuracy,
                   'train_predictions': train_predictions,
                   'val_predictions': val_predictions}
        return results

    def fit_tree(self, X, y, depth=0):
        """Fit a decision tree with recursive splitting on nodes.

        Args:
            X (ndarray): training set without class labels.
            y (ndarray) : class labels for training set.
            depth (int) : starting depth of decision tree.
        Returns:
            tree (Node): root node of learned decision tree.
        """
        # Get number of training observations in current node with each class label 0 and 1.
        class_distribution = [np.sum(y == i) for i in range(self.n_classes)]

        # Instantiate node to grow the decision tree.
        tree = Node(n=y.size,
                    class_distribution=class_distribution,
                    gini_index=_gini(y, self.n_classes))

        # Perform recursive splitting to max depth.
        if depth < self.max_depth:
            gini_index, split_index = self.get_split(X, y)
            # Get indices for data and class labels to go to the left child, send the rest to the right child.
            if split_index is not None:
                index_left = (X[:, split_index] == 1)
                X_left, y_left = X[index_left], y[index_left]
                X_right, y_right = X[~index_left], y[~index_left]
                tree.gini_index = gini_index
                tree.feature_index = split_index
                depth += 1
                tree.left = self.fit_tree(X_left, y_left, depth=depth)
                tree.right = self.fit_tree(X_right, y_right, depth=depth)
        return tree

    def get_split(self, X, y):
        """Find the best split for a node (the split that has the lowest gini impurity).

        Args:
            X (ndarray): training set without class labels.
            y (ndarray) : class labels for training set.
        Returns:
            current_gini (float): gini impurity from best split.
            best_index (int): integer for column index for feature split on.
        """
        # If there are less than two observations at node we don't want to split it.
        if y.size <= 1:
            return None, None

        # Current node impurity prior to splitting.
        current_gini = _gini(y, self.n_classes)

        # Iterate through all features and calculate gini impurity from resulting split.
        split_index = None
        for index in range(self.n_features):
            feature = X[:, index]

            # Get indices to go to left child if feature value is 1, otherwise right child.
            left_idx = (feature == 1)
            left_y = y[left_idx]
            right_y = y[~left_idx]

            # If no observations in left child, set gini index to 0.
            if len(left_y) == 0:
                gini_left = 0
            else:
                gini_left = _gini(left_y, self.n_classes)

            # If no observations in right child, set gini index to 0.
            if len(right_y) == 0:
                gini_right = 0
            else:
                gini_right = _gini(right_y, self.n_classes)

            # Calculate probabilities for left and right children.
            prob_left = len(left_y) / len(y)
            prob_right = len(right_y) / len(y)

            # Gini index from split is just weighted average of gini indices from left and right children.
            split_gini = prob_left * gini_left + prob_right * gini_right

            # Update current gini value if the split is beneficial, save feature index associated with split.
            if split_gini < current_gini:
                current_gini = split_gini
                split_index = index
        return current_gini, split_index
