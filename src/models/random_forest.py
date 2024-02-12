import numpy as np
from scipy.stats import mode
from src.models.cart import CART
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, 
                 min_samples_split=2, min_impurity_decrease=0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.trees_ = Parallel(n_jobs=-1)(
            delayed(self._fit_tree)(X, y) for _ in range(self.n_estimators)
        )
        return self  # Important for Scikit-learn compatibility
            
    def _fit_tree(self, X, y):
        n_samples, n_features = X.shape
        if self.bootstrap:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
        else:
            X_sample, y_sample = X, y
        
        if isinstance(self.max_features, str):
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
        
        tree = CART(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease, max_features=max_features)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        # Validate input
        X = check_array(X)
        check_is_fitted(self, 'trees_')
        
        # Collect predictions from each tree and take the mode
        tree_preds = np.array([tree.predict(X) for tree in self.trees_])
        mode_preds, _ = mode(tree_preds, axis=0)
        return mode_preds

