import numpy as np

class DecisionStump:
    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_polarity = None

        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples) * -1
                    if polarity == 1:
                        predictions[feature_values < threshold] = 1
                    else:
                        predictions[feature_values > threshold] = 1
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        best_feature = feature
                        best_threshold = threshold
                        best_polarity = polarity

        self.feature = best_feature
        self.threshold = best_threshold
        self.polarity = best_polarity

    def predict(self, X):
        feature_values = X[:, self.feature]
        predictions = np.ones(X.shape[0]) * -1
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = 1
        else:
            predictions[feature_values > self.threshold] = 1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)
            error = np.sum(weights[predictions != y])
            if error == 0:
                alpha = 1.0
            else:
                alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.stumps.append(stump)
            # update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            final_pred += alpha * stump.predict(X)
        return np.sign(final_pred)

    def predict_at_iteration(self, X, iteration):
        final_pred = np.zeros(X.shape[0])
        for i in range(min(iteration, len(self.alphas))):
            final_pred += self.alphas[i] * self.stumps[i].predict(X)
        return np.sign(final_pred)