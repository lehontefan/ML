import numpy as np
from mylib.GradientDescent import gradientDescent

class LogisticRegression:
    def __init__(self, iterations = 1000, learning_rate = 0.01, lambda_reg = 1, regularization = 'L2', eta = 10**-6, bias=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = []
        self.lambda_reg = lambda_reg
        self.regularization = regularization
        self.eta = eta
        self.k = None
        self.bias = bias

    def fit(self, X, y):
        self.k = len(np.unique(y))
        if self.bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = gradientDescent(X, y, iterations=self.iterations, learning_rate=self.learning_rate, eta=self.eta, loss='log'
                                       , regularization=self.regularization, lambda_reg=self.lambda_reg)

    def predict(self, X):
        if self.bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return 1 / (1 + np.exp(-np.dot(X, self.weights))) if self.k < 3 else np.exp(np.dot(X, self.weights)) / np.sum(np.exp(np.dot(X, self.weights)), axis=1, keepdims=True)