import numpy as np

def cosine(v, w):
    v = np.array(v)
    w = np.array(w)
    return np.dot(v, w) / (np.linalg.norm(v)*np.linalg.norm(w))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Skipgram:
    def __init__(self, learning_rate=0.01, window=2, size=100, neg_per_pos=2, alpha=0.75):
        self.learning_rate = learning_rate
        self.window = window
        self.size = size
        self.neg_per_pos = neg_per_pos
        self.vocabulary_size = None
        self.vocabulary = None
        self.w_target = {}
        self.w_context = {}
        self.alpha = alpha

    def fit(self, text, n_epochs=5):
        self.vocabulary = list(np.unique(text))
        self.vocabulary_size = len(self.vocabulary)
        for word in self.vocabulary:
            self.w_target[word] = np.random.uniform(-0.1, 0.1, self.size)
            self.w_context[word] = np.random.uniform(-0.1, 0.1, self.size)
        words_arr, counts = np.unique(text, return_counts=True)
        freq = counts.astype(float) ** self.alpha
        freq /= freq.sum()
        cumulative = np.cumsum(freq)
        for epoch in range(n_epochs):
            for index in range(self.window, len(text) - self.window):
                pos_context = text[index - self.window:index] + text[index + 1:index + self.window + 1]
                exclude = set(pos_context)
                exclude.add(text[index])
                neg_context = []
                while len(neg_context) < 2 * self.window * self.neg_per_pos:
                    r = np.random.random(4 * self.window * self.neg_per_pos)
                    idx = np.searchsorted(cumulative, r)
                    batch = words_arr[idx]
                    neg_context.extend([w for w in batch if w not in exclude])
                neg_context = neg_context[:2 * self.window * self.neg_per_pos]
                target_dl = np.zeros(self.size)
                for i in range(2 * self.window):
                    target_dl += self.learning_rate * self.w_context[pos_context[i]] * (
                                sigmoid(np.dot(self.w_context[pos_context[i]], self.w_target[text[index]])) - 1)
                    self.w_context[pos_context[i]] -= self.learning_rate * self.w_target[text[index]] * (
                                sigmoid(np.dot(self.w_context[pos_context[i]], self.w_target[text[index]])) - 1)
                    for j in range(self.neg_per_pos):
                        target_dl += self.learning_rate * self.w_context[neg_context[i * self.neg_per_pos + j]] * (
                            sigmoid(np.dot(self.w_context[neg_context[i * self.neg_per_pos + j]],
                                           self.w_target[text[index]])))
                        self.w_context[neg_context[i * self.neg_per_pos + j]] -= self.learning_rate * self.w_target[
                            text[index]] * (sigmoid(
                            np.dot(self.w_context[neg_context[i * self.neg_per_pos + j]], self.w_target[text[index]])))
                self.w_target[text[index]] -= target_dl

    def get_similar(self, word, n=5):
        if word not in self.w_target:
            return []
        vec = self.w_target[word]
        scores = {}
        for w, v in self.w_target.items():
            if w == word:
                continue
            scores[w] = np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v))
        return sorted(scores.items(), key=lambda x: -x[1])[:n]