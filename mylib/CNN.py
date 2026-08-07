from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class CNN:
    def __init__(self, input_size, n_filters, filter_size, filter_step, n_classes):
        self.input_size = input_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filter_step = filter_step
        self.n_classes = n_classes
        self.feature_map_size = int(((input_size - filter_size) / filter_step + 1))
        self.filter_weights = np.random.uniform(-0.1, 0.1, size=(n_filters, filter_size, filter_size))
        self.ffn_in_weights = np.random.uniform(-0.1, 0.1, size=(self.feature_map_size ** 2 * self.n_filters,
                                                                 self.feature_map_size ** 2 * self.n_filters * 4))
        self.ffn_out_weights = np.random.uniform(-0.1, 0.1,
                                                 size=(self.feature_map_size ** 2 * self.n_filters * 4, n_classes))

    def backward(self, X, y, learning_rate=0.01, iterations=100):
        batch_size = X.shape[0]
        for iteration in range(iterations):
            y_pred, cache = self.forward(X)
            delta = y_pred - y
            d_out_weights = learning_rate * np.dot(cache['activation'].T, delta) / batch_size
            delta = np.dot(delta, self.ffn_out_weights.T) * (cache['hidden_layer'] > 0)
            self.ffn_out_weights -= d_out_weights
            d_in_weights = learning_rate * np.dot(cache['norm'].T, delta) / batch_size
            delta = np.dot(delta, self.ffn_in_weights.T)
            self.ffn_in_weights -= d_in_weights
            delta = (1 / cache['logits_std']) * (
                        delta - delta.mean(axis=-1, keepdims=True) - (cache['logits'] - cache['logits_mean']) / cache[
                    'logits_std'] * ((cache['logits'] - cache['logits_mean']) / cache['logits_std'] * delta).mean(
                    axis=-1, keepdims=True))
            d_filter_weights = learning_rate * np.einsum('i f r c, i r c p q -> f p q',
                                                         delta.reshape(X.shape[0], self.n_filters,
                                                                       self.feature_map_size, self.feature_map_size),
                                                         cache['input_X']) / batch_size
            self.filter_weights -= d_filter_weights
            if iteration % 10 == 0:
                print(iteration, '%')

    def forward(self, X):
        cache = {}
        input_X = sliding_window_view(X.reshape(X.shape[0], -1, self.input_size), (self.filter_size, self.filter_size),
                                      axis=(1, 2))[:, ::self.filter_step, ::self.filter_step].reshape(X.shape[0], -1,
                                                                                                      self.filter_size,
                                                                                                      self.filter_size)
        cache['input_X'] = input_X.reshape(X.shape[0], self.feature_map_size, self.feature_map_size, self.filter_size,
                                           -1).copy()
        logits = np.einsum('i r c p q, f p q -> i f r c',
                           input_X.reshape(X.shape[0], self.feature_map_size, self.feature_map_size, self.filter_size,
                                           -1), self.filter_weights).reshape(X.shape[0],
                                                                             self.feature_map_size ** 2 * self.n_filters)
        cache['logits'] = logits.copy()
        cache['logits_mean'] = logits.mean(axis=-1, keepdims=True)
        cache['logits_std'] = logits.std(axis=-1, keepdims=True)
        norm = (logits - logits.mean(axis=-1, keepdims=True)) / logits.std(axis=-1, keepdims=True)
        cache['norm'] = norm.copy()
        hidden_layer = np.dot(norm, self.ffn_in_weights)
        cache['hidden_layer'] = hidden_layer.copy()
        activation = np.maximum(0, hidden_layer)
        cache['activation'] = activation.copy()
        output_layer = np.dot(activation, self.ffn_out_weights)
        output = softmax(output_layer)
        return output, cache