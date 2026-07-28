import numpy as np
from mylib.GradientDescent import gradientDescent

class LinearRegression:
    def __init__(self, iterations = 1000, learning_rate = 0.01, lambda_reg = 1, regularization = 'L2', eta = 10**-6):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = []
        self.lambda_reg = lambda_reg
        self.regularization = regularization
        self.eta = eta

    def fit(self, X, y):
        self.weights = gradientDescent(X, y, iterations=self.iterations, learning_rate=self.learning_rate, eta=self.eta, loss='MSE'
                                       , regularization=self.regularization, lambda_reg=self.lambda_reg)

    def predict(self, X):
        return np.dot(X, self.weights)