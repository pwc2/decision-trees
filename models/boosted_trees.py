"""
    File name: boosted_trees.py
    Author: Patrick Cummings
    Date created: 11/18/2019
    Date last modified: 11/18/2019
    Python Version: 3.7

"""

import numpy as np

from models.functions import _gini, _predict, _predict_adaboost, _accuracy
from models.node import Node


class BoostedTrees:
    """Class to construct decision tree object with AdaBoost.

    """

    def __init__(self, train, validation, test, label, n_classifiers, max_depth=None):
        """Constructs BoostedTrees object.

        Args:
            train (DataFrame): DataFrame with training set observations.
            validation (DataFrame): DataFrame with validation set observations.
            test (DataFrame): DataFrame with test set observations.
            label (str): String with column name for class labels in train and validation sets.
            n_classifiers (int): Number of base classifiers to use.
            max_depth (int): Maximum depth to grow decision tree to.

        Returns:
            None
        """
        # Extract features and labels from data sets
        self.train_features = train.drop(label, axis=1)
        self.train_labels = train[label]
        self.validation_features = validation.drop(label, axis=1)
        self.validation_labels = validation[label]
        self.test_features = test

        self.n_classes = len(set(self.train_labels))
        self.n_features = np.size(self.train_features.to_numpy(), axis=1)
        self.n_samples = np.size(self.train_features.to_numpy(), axis=0)
        self.n_classifiers = n_classifiers
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
        """Learns ensemble of boosted decision trees (stumps if max_depth=1) from training data and generate
        predictions and accuracy on training and validation sets.

        Returns:
            results (dict): dictionary with predictions and accuracy for training and validation sets.
        """
        X_train = self.train_features.to_numpy()
        y_train = self.train_labels.to_numpy()
        X_val = self.validation_features.to_numpy()
        y_val = self.validation_labels.to_numpy()

        # Lists to store base classifiers and weights for each base classifier.
        L = self.n_classifiers
        base_classifiers = []
        base_classifier_alphas = []

        # Weights for training instances, initialize as 1/N.
        N = self.n_samples
        D = np.ones(N) / N

        # Iterate for each t to generate weak learner to add to ensemble.
        for t in range(L):
            # Learn decision tree.
            self.tree = self.fit_tree(X_train, y_train, weights=D, depth=0)

            # Compute the error (epsilon) and alpha on the predictions.
            predictions = np.array(self.predict(X_train))
            incorrect = (predictions != y_train)
            epsilon = np.mean(np.average(incorrect, weights=D, axis=0))
            alpha = np.log((1 - epsilon) / epsilon) / 2

            # Create copy of labels and predictions and change class labels to {-1, 1} so weights update properly below.
            _y_train = np.array([-1 if y == 0 else 1 for y in y_train])
            _predictions = np.array([-1 if y == 0 else 1 for y in predictions])

            # Update weights for the next classifier and normalize.
            D = D * np.exp(-alpha * _y_train * _predictions)
            D = D / D.sum()

            # Save alpha and weak learner for weighted predictions.
            base_classifiers.append(self.tree)
            base_classifier_alphas.append(alpha)

        # Calculate predictions and accuracy on train and validation sets.
        train_predictions = _predict_adaboost(base_classifiers, base_classifier_alphas, X_train)
        val_predictions = _predict_adaboost(base_classifiers, base_classifier_alphas, X_val)
        train_accuracy = _accuracy(train_predictions, y_train)
        val_accuracy = _accuracy(val_predictions, y_val)

        results = {'max_depth': self.max_depth,
                   'n_classifiers': self.n_classifiers,
                   'train_accuracy': train_accuracy,
                   'val_accuracy': val_accuracy,
                   'train_predictions': train_predictions,
                   'val_predictions': val_predictions}
        return results

    def fit_tree(self, X, y, weights, depth=0):
        """Fit a decision tree with recursive splitting on nodes, takes additional weight argument for AdaBoost.

        Args:
            X (ndarray): training set without class labels.
            y (ndarray): class labels for training set.
            weights (ndarray): weights for each training instance.
            depth (int): starting depth of decision tree.
        Returns:
            tree (Node): root node of learned decision tree.
        """
        # Get sum of weights from each class (0 and 1) in current node.
        D = weights
        class_weights = [np.sum(D * (y == i)) for i in range(self.n_classes)]

        # Instantiate node to grow the decision tree.
        tree = Node(n=y.size,
                    class_distribution=class_weights,
                    gini_index=_gini(y, self.n_classes, weights=D))

        # Perform recursive splitting to max depth.
        if depth < self.max_depth:
            gini_index, split_index = self.get_split(X, y, weights=D)
            # Get indices for data, class labels, and weights to go to the left child, send the rest to the right child.
            if split_index is not None:
                index_left = (X[:, split_index] == 1)
                X_left, y_left, D_left = X[index_left], y[index_left], D[index_left]
                X_right, y_right, D_right = X[~index_left], y[~index_left], D[~index_left]
                tree.gini_index = gini_index
                tree.feature_index = split_index
                depth += 1
                tree.left = self.fit_tree(X_left, y_left, weights=D_left, depth=depth)
                tree.right = self.fit_tree(X_right, y_right, weights=D_right, depth=depth)
        return tree

    def get_split(self, X, y, weights):
        """Find the best split for a node (the split that has the lowest gini impurity), takes additional weight
        argument for AdaBoost.

        Args:
            X (ndarray): training set without class labels.
            y (ndarray): class labels for training set.
            weights (ndarray): weights for each training instance.
        Returns:
            current_gini (float): gini impurity from best split.
            best_index (int): integer for column index for feature split on.
        """
        # If there are less than two instances at node we don't want to split it.
        if y.size <= 1:
            return None, None

        # Get sum of weights from each class (0 and 1) in current node.
        D = weights
        class_weights = [np.sum(D * (y == i)) for i in range(self.n_classes)]

        # Current node impurity prior to splitting.
        current_gini = _gini(y, self.n_classes, weights=D)

        # Iterate through all features and calculate gini impurity from resulting split.
        split_index = None
        for index in range(self.n_features):
            feature = X[:, index]

            # Get indices to go to left child if feature value is 1, otherwise right child.
            left_idx = (feature == 1)
            left_y = y[left_idx]
            left_D = D[left_idx]
            right_y = y[~left_idx]
            right_D = D[~left_idx]

            # If no observations in left child, set gini index to 0.
            if len(left_y) == 0:
                left_class_weights = [0, 0]
                gini_left = 0
            else:
                left_class_weights = [np.sum(left_D * (left_y == i)) for i in range(self.n_classes)]
                gini_left = _gini(left_y, self.n_classes, weights=left_D)

            # If no observations in right child, set gini index to 0.
            if len(right_y) == 0:
                right_class_weights = [0, 0]
                gini_right = 0
            else:
                right_class_weights = [np.sum(right_D * (right_y == i)) for i in range(self.n_classes)]
                gini_right = _gini(right_y, self.n_classes, weights=right_D)

            # Calculate probabilities for left and right children.
            prob_left = sum(left_class_weights) / sum(class_weights)
            prob_right = sum(right_class_weights) / sum(class_weights)

            # Gini index from split is just weighted average of gini indices from left and right children.
            split_gini = prob_left * gini_left + prob_right * gini_right

            # Update current gini value if the split is beneficial, save feature index associated with split.
            if split_gini < current_gini:
                current_gini = split_gini
                split_index = index
        return current_gini, split_index
