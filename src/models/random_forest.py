import numpy as np
from scipy.stats import mode
from src.models.cart import CART
from joblib import Parallel, delayed

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, 
                 min_samples_split=2, min_impurity_decrease=0, bootstrap=True):
        self.n_estimators = n_estimators  # Number of trees
        self.max_features = max_features  # The number of features to consider when looking for the best split
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap  # Whether bootstrap samples are used when building trees
        self.trees = []  # List to store all fitted tree models

    def fit(self, X, y):
        self.trees = []
        
        # Parallelize tree fitting
        self.trees = Parallel(n_jobs=-1)(delayed(self._fit_tree)(X, y) for _ in range(self.n_estimators))
        
            
    def _fit_tree(self, X, y):
        
        n_samples, n_features = X.shape
        # Bootstrap sampling
        if self.bootstrap:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
        else:
            X_sample, y_sample = X, y
        
        # Handle max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
        
        # Initialize and fit a tree model
        tree = CART(max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_features=max_features)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        # Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # For classification, take the mode (most common label) across trees
        mode_preds, _ = mode(tree_preds, axis=0)
        return mode_preds


    def predict_proba(self, X):
        # Check if trees have been fitted
        if not self.trees:
            raise ValueError("RandomForestClassifier has not been fitted yet.")

        # Collect probability predictions from each tree
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        
        # Average the probabilities across all trees for each class
        avg_probas = np.mean(tree_probas, axis=0)
        return avg_probas
