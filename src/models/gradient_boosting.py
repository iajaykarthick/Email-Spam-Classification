import numpy as np
from src.models.cart import CART

from src.evaluation.classification_metrics import ClassificationMetrics

class GradientBoostingBinaryClassifier:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_odds(self, p):
        return np.log(p / (1 - p))

    def fit(self, X, y):
        # Initialize predictions to the mean of the target
        p = np.mean(y)
        self.initial_prediction = self._log_odds(p)
        Fm = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # Calculate pseudo-residuals
            p = self._sigmoid(Fm)
            residuals = y - p
            
            # Fit a tree to the pseudo-residuals
            tree = CART(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion='mse')
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update the model with the predictions of the new tree
            y_pred, y_pred_proba = tree.predict(X)
            Fm += self.learning_rate * y_pred
            
    def predict_proba(self, X):
        # Calculate initial predictions
        Fm = np.full(X.shape[0], self.initial_prediction)
        
        # Add the predictions from each tree
        for tree in self.trees:
            y_pred, y_pred_proba = tree.predict(X)
            Fm += self.learning_rate * y_pred
            
        # Calculate probabilities using the sigmoid function
        probs = self._sigmoid(Fm)
        return np.vstack((1 - probs, probs)).T
    
    def predict(self, X):
        # Predict class labels for samples in X
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X, y):
        # Evaluate the model using the given data
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        metrics = ClassificationMetrics(y, y_pred, y_pred_proba[:, 1])
        return metrics
