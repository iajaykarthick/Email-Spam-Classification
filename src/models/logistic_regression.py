import numpy as np

from src.evaluation.classification_metrics import ClassificationMetrics

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, tolerance=1e-4, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.lambda_reg = lambda_reg

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        previous_loss = float('inf')

        for i in range(self.num_iterations):
            # Calculate predictions
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute the cost with epsilon to avoid log(0)
            epsilon = 1e-5
            regularization_term = (self.lambda_reg / (2 * len(y))) * np.sum(np.square(self.weights))
            cost = (-1 / len(y)) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon)) + regularization_term

            # Check for convergence
            if previous_loss - cost < self.tolerance:
                print(f"Convergence reached at iteration {i}.")
                break
            previous_loss = cost

            # Compute gradients with regularization (excluding bias term from regularization)
            dw = (1 / len(y)) * np.dot(X.T, (predictions - y)) + (self.lambda_reg / len(y)) * self.weights
            db = (1 / len(y)) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        print(f"Final loss: {cost}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in predictions]

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        proba_class_1 = self.sigmoid(z)
        proba_class_0 = 1 - proba_class_1
        return np.vstack((proba_class_0, proba_class_1)).T
    