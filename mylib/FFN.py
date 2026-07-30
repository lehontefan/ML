import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

class FFNClassifier:
    def __init__(self, n_input, n_output, n_layers, n_neurons, learning_rate=0.01, iterations=1000):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_output = n_output
        self.n_input = n_input
        self.input_weights = np.random.uniform(-0.1, 0.1, (self.n_neurons, self.n_input))
        self.output_weights = np.random.uniform(-0.1, 0.1, (self.n_output, self.n_neurons))
        if self.n_layers > 1:
            self.hidden_weights = np.random.uniform(-0.1, 0.1, (self.n_neurons * (self.n_layers - 1), self.n_neurons))
        self.learning_rate = learning_rate
        self.iterations = iterations

    def _forward(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z_list = []
        a_list = [X]
        output = np.dot(X, self.input_weights.T)
        z_list.append(output)
        output = ReLU(output)
        a_list.append(output)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                output = np.dot(output, self.hidden_weights.T[:, i * self.n_neurons:i * self.n_neurons + self.n_neurons])
                z_list.append(output)
                output = ReLU(output)
                a_list.append(output)
        output = np.dot(output, self.output_weights.T)
        for i in range(X.shape[0]):
            output[i] = softmax(output[i])
        return output, z_list, a_list

    def fit(self, X, y):
        iteration = 0
        X = np.array(X)
        y = np.array(y)
        unique_vals, y_encoded = np.unique(y, return_inverse=True)
        k = len(unique_vals)
        y_onehot = np.eye(k)[y_encoded]
        while iteration < self.iterations:
            y_pred, z_list, a_list = self._forward(X)
            delta = y_pred - y_onehot
            a_last = a_list[-1]
            d_output_weights = np.dot(delta.T, a_last) / X.shape[0]
            delta = np.dot(delta, self.output_weights) * (z_list[-1] > 0)
            d_hidden_weights = None
            if self.n_layers > 1:
                d_hidden_weights = np.zeros_like(self.hidden_weights)
                for i in reversed(range(self.n_layers - 1)):
                    a_in = a_list[i]
                    W_i = self.hidden_weights[i * self.n_neurons:(i + 1) * self.n_neurons, :]
                    d_hidden_weights[i * self.n_neurons:(i + 1) * self.n_neurons, :] = np.dot(delta.T, a_in) / X.shape[0]
                    delta = np.dot(delta, W_i) * (z_list[i] > 0)
            d_input_weights = np.dot(delta.T, X) / X.shape[0]
            self.output_weights -= self.learning_rate * d_output_weights
            self.input_weights -= self.learning_rate * d_input_weights
            if d_hidden_weights is not None:
                self.hidden_weights -= self.learning_rate * d_hidden_weights
            iteration += 1

    def predict(self, X):
        output, _, _ = self._forward(X)
        return output