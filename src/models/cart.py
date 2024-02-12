import numpy as np


class CART:
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, num_samples=None, class_distribution=None, value=None):
            self.feature = feature # feature index to split
            self.threshold = threshold # threshold to split
            self.left = left # left node
            self.right = right # right node
            self.num_samples = num_samples # number of samples in the node
            self.class_distribution = class_distribution # class distribution of samples in the node
            self.value = value # value of the node if it is a leaf node
            
        def is_leaf(self):
            return self.value is not None
        
        def __repr__(self):
            if self.is_leaf():
                return f"Leaf: {self.value}"
            return f"Node: feature={self.feature}, threshold={self.threshold}"
    
    
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0, max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
        
        # for progress tracking
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
        if (self.max_depth is None or depth < self.max_depth) and num_samples >= self.min_samples_split:
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
                    right=right_node,
                    num_samples=num_samples,
                    class_distribution=self._class_distribution(y)
                )
                return current_node
        
        # leaf node
        if len(y) == 0:
            raise ValueError("No samples in the node. This should not happen.")
        leaf_value = self._to_leaf(y)
        return self.Node(
            num_samples=num_samples, 
            class_distribution=self._class_distribution(y),
            value=leaf_value
        )
        
    def _to_leaf(self, y):
        if self.criterion == 'mse':
            return np.mean(y)
        return self._bincount(y).argmax()
        
    def _class_distribution(self, y):
        if self.criterion == 'mse':
            return None
        return self._bincount(y)
    
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
    
    def _bincount(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return np.array([0 if i not in unique else counts[unique == i][0] for i in range(max(unique) + 1)])
    
    def _gini(self, y):
        probabilities = self._bincount(y) / len(y)
        gini = 1 - np.sum([p**2 for p in probabilities if p > 0])
        return gini
    
    def _entropy(self, y):
        # y should be non-negative integer labels
        probabilities = self._bincount(y) / len(y)
        entropy = np.sum([p * -np.log2(p) for p in probabilities if p > 0])
        return entropy
    
    # Regression - MSE
    def _mse(self, y):
        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        return mse
    
    def _information_gain(self, y, left_child, right_child):
        weight_1 = len(left_child) / len(y)
        weight_2 = len(right_child) / len(y)
        
        if self.criterion == 'gini':
            parent_impurity = self._gini(y)
            left_impurity = self._gini(left_child)
            right_impurity = self._gini(right_child)
        
        elif self.criterion == 'mse':
            parent_impurity = self._mse(y)
            left_impurity = self._mse(left_child)
            right_impurity = self._mse(right_child)
        
        else:
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(left_child)
            right_impurity = self._entropy(right_child) 
        
        gain = parent_impurity - (weight_1 * left_impurity + weight_2 * right_impurity)
        
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

    def export_graphviz(self, full_verbose=False, leaf_verbose=False):
        
        # Initialize the unique ID generator
        unique_id_generator = self._unique_id_generator()
        
        # Start the dot string with the graph type
        dot_string = "digraph Tree {\n size=\"10,10\"; rankdir=\"LR\";\n"
    
        # Begin the recursive process starting from the root node
        dot_string, root_id = self._export_node(self.root, dot_string, unique_id_generator, full_verbose=full_verbose, leaf_verbose=leaf_verbose)

        # Close the graph string
        dot_string += "}\n"

        return dot_string

    def _unique_id_generator(self):
        current_id = 0
        while True:
            yield current_id
            current_id += 1

    def _export_node(self, node, dot_string, unique_id_generator, full_verbose=False, leaf_verbose=False):
        # Get a unique ID for the current node
        unique_id = next(unique_id_generator)
        if node.is_leaf():
            
            # Leaf node definition with value
            if full_verbose or leaf_verbose:
                dot_string += f"  {unique_id} [shape=box, label=\"Predicted class: {node.value} \\n samples = {node.num_samples}\\n class distribution = {node.class_distribution}\"];\n"
            else:
                dot_string += f"  {unique_id} [shape=box, label=\"Predicted class: {node.value}\"];\n"
            
        
        else:
            
            # Decision node definition
            if full_verbose:
                dot_string += f"  {unique_id} [label=\"X[{node.feature}] <= {node.threshold:.3f} \\n samples = {node.num_samples}\\n class distribution = {node.class_distribution}\"];\n"
            else:
                dot_string += f"  {unique_id} [label=\"X[{node.feature}] <= {node.threshold:.3f}\"];\n"            
            
            # Recursively process the left child
            dot_string, left_child_id = self._export_node(node.left, dot_string, unique_id_generator, full_verbose=full_verbose, leaf_verbose=leaf_verbose)
            
            # Add edge to the left child
            dot_string += f"  {unique_id} -> {left_child_id} [label=\"true\"];\n"
            
            # Recursively process the right child
            dot_string, right_child_id = self._export_node(node.right, dot_string, unique_id_generator, full_verbose=full_verbose, leaf_verbose=leaf_verbose)
            
            # Add edge to the right child
            dot_string += f"  {unique_id} -> {right_child_id} [label=\"false\"];\n"

        return dot_string, unique_id
