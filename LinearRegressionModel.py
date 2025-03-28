import numpy as np

class LinearRegression:
    def __init__(self,layers,lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.bias = 0
        self.layers = layers
    def fit(self,X,y):
        n_samples = len(y)
        for _ in range(self.epochs):
            y_predicted = X * self.w + self.bias
            error = y_predicted - y
            dw = (2/n_samples) * np.sum(X * error)
            db = (2/n_samples) * np.sum(error)
            self.w -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self,X):
        return X * self.w + self.bias
