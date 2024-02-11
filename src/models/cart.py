import numpy as np


class CART:
    """
        CART works by repeatedly partitioning the data into subsets based on the feature that results in 
        the highest information gain (IG) or the lowest Gini impurity for classification, 
        and the lowest mean squared error (MSE) or mean absolute error (MAE) for regression. 
        This process is recursively applied to each subset until a stopping criterion is met 
        (e.g., maximum depth of the tree, minimum samples in a node, or no further improvement).
        
        ### For Classification

        - **Gini Impurity**: 
            A measure of how often a randomly chosen element from the set would be incorrectly labeled 
            if it was randomly labeled according to the distribution of labels in the subset.
            The Gini impurity of a dataset is:

            $$ Gini = 1 - \sum_{i=1}^{n} p_i^2 $$

            where $p_i$ is the proportion of items labeled with class $i$ in the dataset.
        
        - **Information Gain**: The change in entropy after the dataset is split on an attribute. It's used to decide which feature to split on at each step in building the tree.

            $$ IG(D, a) = Entropy(D) - \sum_{v \in Values(a)} \frac{|D_v|}{|D|} Entropy(D_v) $$

            where $Entropy(D)$ is the entropy of the dataset $D$, $Values(a)$ are the unique values of attribute $a$, and $D_v$ is the subset of $D$ for which attribute $a$ has value $v$.
            
        ### Pre-pruning
        - `max_depth`: Stop tree growth after reaching a specified depth.

        - `min_samples_split`: Don't split nodes if fewer than a set number of samples are present.

        - `min_impurity_decrease`: Only split nodes if a minimum impurity reduction is achieved.

        - `max_features`: Considers only a random subset of features at each split (similar to Random Forests). This introduces more randomness and helps in diversity and reduce overfitting.
    """
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature # feature index to split
            self.threshold = threshold # threshold to split
            self.left = left # left node
            self.right = right # right node
            self.value = value # value of the node if it is a leaf node
            
        def is_leaf(self):
            return self.value is not None
        
        def __repr__(self):
            if self.is_leaf():
                return f"Leaf: {self.value}"
            return f"Node: feature={self.feature}, threshold={self.threshold}"
    
    
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.root = None
        
        # for progress tracking
        self.current_depth = 0
        self.max_reached_depth = 0
        
    def fit(self, X, y, verbose=False):
        self.max_reached_depth = 0
        self.root = self._build_tree(X, y, depth=0)
        if verbose:
            print(f"Maximum depth reached during fit: {self.max_reached_depth}")
        
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        if depth > self.max_reached_depth:
            self.max_reached_depth = depth
        
        # check if stopping criteria are not met
        if depth < self.max_depth and num_samples >= self.min_samples_split:
            # find the best split
            best_split = self._best_split(X, y, num_samples, num_features)  
            # if the gain is greater than the minimum impurity decrease
            if best_split.get('gain', -1) >= self.min_impurity_decrease:
                if len(y[best_split['left_indices']]) == 0 or len(y[best_split['right_indices']]) == 0:
                    raise ValueError("Left or right indices are empty. This should not happen.")
                left_node = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
                right_node = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
                current_node = self.Node(
                    feature=best_split['feature_index'],
                    threshold=best_split['threshold'],
                    left=left_node,
                    right=right_node
                )
                return current_node
        
        # leaf node
        if len(y) == 0:
            raise ValueError("No samples in the node. This should not happen.")
        leaf_value = self._to_leaf(y)
        return self.Node(value=leaf_value)
        
    def _to_leaf(self, y):
        # majority class 
        # y should be non-negative integer labels
        return np.bincount(y).argmax()
    
    def _best_split(self, X, y, num_samples, num_features):
        best_split = {}
        max_gain = -float('inf')
        
        if self.max_features is not None:
            num_features_to_sample = min(self.max_features, num_features) # select a subset of features to split
            possible_feature_indices = np.random.choice(num_features, num_features_to_sample, replace=False)
        
        else:
            possible_feature_indices = range(num_features)
        
        for feature_index in possible_feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_indices, right_indices = self._split_data(X, y, feature_index, threshold)
                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_y = y[left_indices]
                    right_y = y[right_indices]
                    gain = self._information_gain(y, left_y, right_y)
                    if gain > max_gain:
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'left_indices': left_indices,
                            'right_indices': right_indices,
                            'gain': gain
                        }
                        max_gain = gain
        
        return best_split
    
    def _split_data(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices
    
    
    def _entropy(self, y):
        # y should be non-negative integer labels
        probabilities = np.bincount(y) / len(y)
        entropy = np.sum([p * -np.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def _information_gain(self, y, left_child, right_child):
        weight_1 = len(left_child) / len(y)
        weight_2 = len(right_child) / len(y)
        
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_child)
        right_entropy = self._entropy(right_child)
        
        gain = parent_entropy - (weight_1 * left_entropy + weight_2 * right_entropy)
        
        return gain
    
    def predict(self, X):
        predictions = [self._predict_input(x, self.root) for x in X]
        return np.array(predictions)
    
    def _predict_input(self, x, node):
        if node.is_leaf():
            return node.value
        feature_value = x[node.feature]
        if feature_value <= node.threshold:
            return self._predict_input(x, node.left)
        else:
            return self._predict_input(x, node.right)
        