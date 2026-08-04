import numpy as np

def norm(X, eps=1e-8):
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)

def ReLU(x):
    return np.maximum(0, x)


def norm_backward(delta, X, eps=1e-8):
    D = X.shape[-1]
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)  # std**2
    std_inv = 1.0 / np.sqrt(var + eps)
    Y = (X - mean) * std_inv
    sum_delta = np.sum(delta, axis=-1, keepdims=True)
    sum_delta_dot_Y = np.sum(delta * Y, axis=-1, keepdims=True)
    dX = (D * delta - sum_delta - Y * sum_delta_dot_Y) * (std_inv / D)
    return dX

class Transformer:
    def __init__(self, blocks, heads, d_model, d_key, d_value, max_seq_len, vocab_size, iterations=1000, learning_rate=0.01, d_ff=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.blocks = blocks
        self.heads = heads
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.vocab_size = vocab_size
        self.w_q = np.random.uniform(-0.1, 0.1, size=(blocks, d_model, d_key * heads))
        self.w_k = np.random.uniform(-0.1, 0.1, size=(blocks, d_model, d_key * heads))
        self.w_v = np.random.uniform(-0.1, 0.1, size=(blocks, d_model, d_value * heads))
        self.w_o = np.random.uniform(-0.1, 0.1, size=(blocks, d_value * heads, d_model))
        self.w_in = np.random.uniform(-0.1, 0.1, size=(blocks, d_model, self.d_ff))
        self.w_out = np.random.uniform(-0.1, 0.1, size=(blocks, self.d_ff, d_model))
        self.pos_embeddings = np.random.uniform(-0.02, 0.02, size=(max_seq_len, d_model))
        self.w_clf = np.random.uniform(-0.02, 0.02, size=(d_model, vocab_size))

    def _backward(self, X, y):
        batch = X.shape[0] * X.shape[1]
        y_pred, cache = self._forward(X)
        delta = y_pred - y # (  batch_size, seq_len, vocab_size)
        # np.array(cache['new_X'][-1]) - (batch_size, seq_len, d_model)
        d_w_clf = self.learning_rate * np.dot(np.array(cache['new_X'][-1]).reshape(-1, self.d_model).T, delta.reshape(-1, self.vocab_size)) / batch
        delta = np.matmul(delta, self.w_clf.T) # (batch_size, seq_len, d_model)
        self.w_clf -= d_w_clf
        for i in reversed(range(self.blocks)):
            # cache['a'][i] - (batch_size, seq_len, d_ff)
            d_w_out = self.learning_rate * np.dot(cache['a'][i].reshape(-1, self.d_ff).T, delta.reshape(-1, self.d_model)) / batch
            delta_residual = delta.copy()
            # (cache['h'][i] > 0) - (batch_size, seq_len, d_ff)
            delta = np.matmul(delta, self.w_out[i].T) * (cache['h'][i] > 0) # (batch_size, seq_len, d_ff)
            self.w_out[i] -= d_w_out
            # cache['d_X'][i] - (batch_size, seq_len, d_model)
            d_w_in = self.learning_rate * np.dot(cache['d_X'][i].reshape(-1, self.d_model).T, delta.reshape(-1, self.d_ff)) / batch
            delta = np.matmul(delta, self.w_in[i].T) # (batch_size, seq_len, d_model)
            self.w_in[i] -= d_w_in
            delta = norm_backward(delta, cache['new_X_old'][i]) # (batch_size, seq_len, d_model)
            delta += delta_residual
            delta_residual = delta.copy()
            # cache['concat'][i] - (batch_size, seq_len, d_model)
            d_w_o = self.learning_rate * np.dot(cache['concat'][i].reshape(-1, self.heads * self.d_value).T, delta.reshape(-1, self.d_model)) / batch
            delta = np.matmul(delta, self.w_o[i].T) # (batch_size, seq_len, d_model)
            self.w_o[i] -= d_w_o
            dQ = np.zeros((X.shape[0], X.shape[1], self.heads * self.d_key))
            dK = np.zeros((X.shape[0], X.shape[1], self.heads * self.d_key))
            dV = np.zeros((X.shape[0], X.shape[1], self.heads * self.d_value))
            for j in reversed(range(self.heads)):
                dV[:, :, j * self.d_value: (j + 1) * self.d_value] = np.matmul(cache['weights' + str(i)][j].transpose(0, 2, 1), delta[:, :, j * self.d_value : (j + 1) * self.d_value])
                d_weights = np.matmul(delta[:, :, j * self.d_value : (j + 1) * self.d_value], cache['V' + str(i)][j].transpose(0, 2, 1))
                d_scores = cache['weights' + str(i)][j] * (d_weights - np.sum(d_weights * cache['weights' + str(i)][j], axis=-1, keepdims=True)) / np.sqrt(self.d_key)
                dQ[:, :, j * self.d_key: (j + 1) * self.d_key] = np.matmul(d_scores, cache['K' + str(i)][j])
                dK[:, :, j * self.d_key: (j + 1) * self.d_key] = np.matmul(d_scores.transpose(0, 2, 1), cache['Q' + str(i)][j])
            d_w_q = self.learning_rate * np.dot(cache['d_X_old'][i].reshape(-1, self.d_model).T, dQ.reshape(-1, self.heads * self.d_key)) / batch
            d_w_k = self.learning_rate * np.dot(cache['d_X_old'][i].reshape(-1, self.d_model).T, dK.reshape(-1, self.heads * self.d_key)) / batch
            d_w_v = self.learning_rate * np.dot(cache['d_X_old'][i].reshape(-1, self.d_model).T, dV.reshape(-1, self.heads * self.d_value)) / batch
            delta = np.matmul(dQ, self.w_q[i].T) + np.matmul(dK, self.w_k[i].T) + np.matmul(dV, self.w_v[i].T)
            self.w_q[i] -= d_w_q
            self.w_k[i] -= d_w_k
            self.w_v[i] -= d_w_v
            delta = norm_backward(delta, cache['new_X_att'][i])
            delta += delta_residual

    def _forward(self, X):
        seq_len = X.shape[1]
        pos_emb = self.pos_embeddings[:seq_len, :]
        new_X = X + pos_emb
        cache = {}
        for i in range(self.blocks):
            cache.setdefault('new_X_att', []).append(new_X.copy())
            d_X = norm(new_X)
            cache.setdefault('d_X_old', []).append(d_X)
            concat = np.zeros((X.shape[0], seq_len, self.d_value * self.heads))
            Q = np.matmul(d_X, self.w_q[i])
            K = np.matmul(d_X, self.w_k[i])
            V = np.matmul(d_X, self.w_v[i])
            for j in range(self.heads):
                scores = np.matmul(Q[:, :, j * self.d_key:(j + 1) * self.d_key], K[:, :, j * self.d_key:(j + 1) * self.d_key].transpose(0, 2, 1)) / np.sqrt(self.d_key)
                cache.setdefault('Q' + str(i), []).append(Q[:, :, j * self.d_key:(j + 1) * self.d_key])
                cache.setdefault('K' + str(i), []).append(K[:, :, j * self.d_key:(j + 1) * self.d_key])
                mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
                scores[:, mask] = -np.inf
                weights = softmax(scores)
                cache.setdefault('weights' + str(i), []).append(weights)
                head = np.matmul(weights, V[:, :, j * self.d_value:(j + 1) * self.d_value])
                cache.setdefault('V' + str(i), []).append(V[:, :, j * self.d_value:(j + 1) * self.d_value])
                concat[:, :, j * self.d_value:(j + 1) * self.d_value] = head
            cache.setdefault('concat', []).append(concat)
            new_X += np.matmul(concat, self.w_o[i])
            cache.setdefault('new_X_old', []).append(new_X.copy())
            d_X = norm(new_X)
            cache.setdefault('d_X', []).append(d_X)
            h = np.matmul(d_X, self.w_in[i])
            cache.setdefault('h', []).append(h)
            a = ReLU(h)
            cache.setdefault('a', []).append(a)
            o = np.matmul(a, self.w_out[i])
            new_X += o
            cache.setdefault('new_X', []).append(new_X.copy())
        logits = np.matmul(new_X, self.w_clf)
        probs = softmax(logits)
        return probs, cache

    def fit(self, X, y):
        for iteration in range(self.iterations):
            self._backward(X, y)

    def predict(self, X):
        probs, _ = self._forward(X)
        return probs