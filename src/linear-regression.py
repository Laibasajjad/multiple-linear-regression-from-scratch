import numpy as np


class LinearRegressionScratch:
    """
    Multiple Linear Regression implemented from scratch using Gradient Descent.
    """

    def __init__(self, alpha=0.001, epochs=3000):
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.x_mean = None
        self.x_std = None
        self.J_history = []

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    def _scale_features(self, x):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        return (x - self.x_mean) / self.x_std

    # -------------------------------
    # Cost Function
    # -------------------------------
    def _compute_cost(self, x, y):
        m = x.shape[0]
        total_cost = 0

        for i in range(m):
            prediction = np.dot(x[i], self.w) + self.b
            error = prediction - y[i]
            total_cost += error ** 2

        return total_cost / (2 * m)

    # -------------------------------
    # Gradient Computation
    # -------------------------------
    def _compute_gradient(self, x, y):
        m = x.shape[0]
        dj_dw = np.zeros_like(self.w)
        dj_db = 0

        for i in range(m):
            error = np.dot(x[i], self.w) + self.b - y[i]
            dj_dw += error * x[i]
            dj_db += error

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    # -------------------------------
    # Training (Fit Model)
    # -------------------------------
    def fit(self, x, y):
        x_scaled = self._scale_features(x)

        self.w = np.zeros(x.shape[1])
        self.b = 0

        self.J_history = []

        for epoch in range(self.epochs):
            dj_dw, dj_db = self._compute_gradient(x_scaled, y)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            cost = self._compute_cost(x_scaled, y)
            self.J_history.append(cost)

            if epoch % 300 == 0:
                print(f"Epoch {epoch}: Cost {cost:.2f}")

        return self

    # -------------------------------
    # Prediction
    # -------------------------------
    def predict(self, x_new):
        x_new_scaled = (x_new - self.x_mean) / self.x_std
        return np.dot(x_new_scaled, self.w) + self.b