import numpy as np

def gradientDescent(X, y, learning_rate=0.001, iterations=1000, eta=10**-6, loss='MSE', regularization='L2', lambda_reg=1):
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same shape')
    if loss not in ['MSE', 'log']:
        raise ValueError('Invalid loss function')
    if regularization not in ['L2', 'L1', None]:
        raise ValueError('Invalid regularization function')
    iteration = 0
    X = np.array(X)
    y = np.array(y)
    if loss == 'MSE':
        weights = np.zeros(X.shape[1])
    elif loss == 'log' and len(np.unique(y)) < 3:
        k = len(np.unique(y))
        weights = np.zeros(X.shape[1])
        y = np.where(y == y[0], 1, 0)
    elif loss == 'log' and len(np.unique(y)) > 2:
        unique_vals, y_encoded = np.unique(y, return_inverse=True)
        k = len(unique_vals)
        weights = np.zeros([X.shape[1], k])
        y = np.eye(k)[y_encoded]
    weight_distance = float('inf')
    while iteration < iterations and weight_distance > eta:
        y_pred = np.dot(X, weights)
        if loss == 'MSE':
            dL = 2 / X.shape[0] * np.dot(X.T, y_pred - y)
        elif loss == 'log':
            if k < 3:
                y_pred = 1 / (1 + np.exp(-y_pred))
                dL = 1 / X.shape[0] * np.dot(X.T, y_pred - y)
            elif k > 2:
                y_pred = y_pred - np.max(y_pred, axis=1, keepdims=True)
                e_sum = np.sum(np.exp(y_pred), axis=1, keepdims=True)
                y_pred = np.exp(y_pred) / e_sum
                dL = 1 / X.shape[0] * np.dot(X.T, y_pred - y)
        if regularization == 'L1':
            dL += lambda_reg / X.shape[0] * np.sign(weights)
        elif regularization == 'L2':
            dL += 2 * lambda_reg / X.shape[0] * weights
        new_weights = weights - learning_rate * dL
        weight_distance = np.linalg.norm(new_weights - weights, ord=2)
        weights = new_weights
        iteration += 1
    return weights