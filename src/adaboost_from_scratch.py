import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []  # α_k values
        self.stumps = []  # h_k weak learners

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Step 1: Initialize weights
        w = np.ones(n_samples) / n_samples  # w_n^(1) = 1/N
        
        for k in range(self.n_estimators):
            # Step 2a: Train weak learner on weighted data
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            predictions = stump.predict(X)
            
            # Step 2b: Calculate weighted error ε_k
            misclassified = (predictions != y)
            epsilon = np.sum(w * misclassified) / np.sum(w)
            
            # Avoid division by zero or log(0)
            epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)
            
            # Step 2c: Compute α_k = 0.5 * ln((1-ε)/ε)
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            
            # Store
            self.alphas.append(alpha)
            self.stumps.append(stump)
            
            # Step 2d: Update weights w_n^(k+1)
            w = w * np.exp(-alpha * y * predictions)
            
            # Step 2e: Normalize weights
            w = w / np.sum(w)

    def predict_at_iteration(self, X, iteration):
        """Predict using only first 'iteration' weak learners"""
        iteration = min(iteration, len(self.stumps))
        
        # H(x) = sign(Σ α_k * h_k(x))
        weighted_sum = np.zeros(X.shape[0])
        for k in range(iteration):
            predictions = self.stumps[k].predict(X)
            weighted_sum += self.alphas[k] * predictions
            
        return np.sign(weighted_sum)
    
    def predict(self, X):
        """Final prediction using all weak learners"""
        return self.predict_at_iteration(X, self.n_estimators)