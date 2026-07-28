import math
from mylib.Tokenizer import byte_tokenize, bpe_tokenize, word_tokenize

class Ngram:
    def __init__(self, n, lambdas=None, space=True, smoothing=1, level='byte'):
        if n <= 0:
            raise ValueError("n must be positive")
        if level not in ['byte', 'word']:
            raise ValueError("level must be byte or word")
        self.level = level
        self.n = n
        self.space = space
        self.smoothing = smoothing
        self.counts = {}
        self.vocabulary = []
        self.merges = None
        if lambdas is None:
            self.lambdas = [1 / n] * n
        elif len(lambdas) != n or not math.isclose(sum(lambdas), 1, rel_tol=1e-9):
            raise ValueError('lambdas must have same length as n')
        else:
            self.lambdas = lambdas

    def fit(self, text, bpe=None):
        if self.level == 'byte':
            if bpe is None:
                sequence = byte_tokenize(text, self.space)
                sequence = [token for word in sequence for token in word]
                self.vocabulary = list(set(sequence))
            else:
                sequence = bpe.tokenize(text)
                sequence = [token for word in sequence for token in word]
                self.merges = bpe.merges
                self.vocabulary = list(bpe.vocabulary)
        if self.level == 'word':
            if bpe is not None:
                raise ValueError("bpe parameter is not supported when level='word'")
            sequence = word_tokenize(text, self.space)
            self.vocabulary = list(set(sequence))
        self.counts[()] = len(sequence)
        for order in range(1, self.n + 1):
            for index in range(len(sequence) - order + 1):
                gram = tuple(sequence[index:index + order])
                self.counts[gram] = self.counts.get(gram, 0) + 1

    def predict(self, sequence, returned='word'):
        if returned not in ['byte', 'word']:
            raise ValueError("return must be byte or word")
        if self.level == 'byte':
            if self.merges is not None:
                new_sequence = bpe_tokenize(sequence, self.merges, self.space)
            else:
                new_sequence = byte_tokenize(sequence, self.space)
            new_sequence = [token for word in new_sequence for token in word]
        if self.level == 'word':
            new_sequence = word_tokenize(sequence, self.space)
        best_probability = -1
        best_token = None
        for token in self.vocabulary:
            probability = 0
            for order in range(1, self.n + 1):
                if order == 1:
                    history = ()
                else:
                    history = tuple(new_sequence[-(order - 1):])
                ngram = history + (token,)
                num = self.counts.get(ngram, 0) + self.smoothing
                denom = self.counts.get(history, 0) + self.smoothing * len(self.vocabulary)
                p = num / denom if denom > 0 else 0.0
                probability += self.lambdas[order - 1] * p
            if probability > best_probability:
                best_probability = probability
                best_token = token
        if self.level == 'byte' and returned == 'word':
            best_token = bytes.fromhex(best_token).decode('utf-8')
        return best_token