from mylib.Tokenizer import byte_tokenize, bpe_tokenize

class Ngram:
    def __init__(self, n, lambdas=None, space=True, smoothing=1):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.space = space
        self.smoothing = smoothing
        self.counts = {}
        self.vocabulary = []
        self.merges = None
        if lambdas is None:
            self.lambdas = [1 / n] * n
        elif len(lambdas) != n or sum(lambdas) != 1:
            raise ValueError('lambdas must have same length as n')
        else:
            self.lambdas = lambdas

    def fit(self, text, bpe=None):
        if bpe is None:
            sequence = byte_tokenize(text, self.space)
            sequence = [token for word in sequence for token in word]
            self.vocabulary = list(set(sequence))
        else:
            sequence = bpe.tokenize(text)
            sequence = [token for word in sequence for token in word]
            self.merges = bpe.merges
            self.vocabulary = list(bpe.vocabulary)
        for order in range(1, self.n + 1):
            for index in range(len(sequence) - order + 1):
                gram = tuple(sequence[index:index + order])
                self.counts[gram] = self.counts.get(gram, 0) + 1

    def predict(self, sequence):
        if self.merges is not None:
            new_sequence = bpe_tokenize(sequence, self.merges, self.space)
        else:
            new_sequence = byte_tokenize(sequence, self.space)
        new_sequence = [token for word in new_sequence for token in word]
        best_probability = -1
        best_token = None
        for token in self.vocabulary:
            probability = 0
            for order in range(1, self.n + 1):
                history = tuple(new_sequence[-order:])
                ngram = history + (token,)
                num = self.counts.get(ngram, 0) + self.smoothing
                denom = self.counts.get(history, 0) + self.smoothing * len(self.vocabulary)
                p = num / denom if denom > 0 else 0.0
                probability += self.lambdas[order - 1] * p
            if probability > best_probability:
                best_probability = probability
                best_token = token
        return best_token