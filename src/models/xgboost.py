import numpy as np
from src.models.cart import CART

class SimplifiedXGBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = 0.0  # Initial prediction will be updated to log(odds)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_odds(self, p):
        return np.log(p / (1 - p))

    def fit(self, X, y):
        # Convert labels to {0, 1}
        y = (y == 1).astype(int)
        
        # Start with an initial prediction of log(odds)
        p = np.mean(y)
        self.initial_prediction = self._log_odds(p)
        F_m = np.full(len(y), self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals as gradient of logistic loss
            preds = self._sigmoid(F_m)
            residuals = y - preds
            
            # Fit a CART to the pseudo-residuals
            tree = CART(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion='mse')
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update model predictions
            update_preds = tree.predict(X)
            F_m += self.learning_rate * update_preds
            
    def predict_proba(self, X):
        # Aggregate predictions from all trees
        F_m = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F_m += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        probs = self._sigmoid(F_m)
        return np.vstack((1 - probs, probs)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        # Convert probabilities to class labels
        return (proba[:, 1] >= 0.5).astype(int)
