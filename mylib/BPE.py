from mylib.Tokenizer import byte_tokenize, bpe_tokenize

class BytePairEncoder:
    def __init__(self, vocabulary=None, iterations=100, space=True):
        if vocabulary is not None and not isinstance(vocabulary, set):
            raise TypeError("vocabulary must be None or set")
        if not isinstance(space, bool):
            raise TypeError("space must be True or False")
        self.vocabulary = vocabulary
        self.iterations = iterations
        self.merges = []
        self.space = space

    def fit(self, sequence):
        if not isinstance(sequence, str):
            raise TypeError("sequence must be a string")
        new_sequence = byte_tokenize(sequence, self.space)
        new_sequence = [word[:] for word in new_sequence]
        iteration = 0
        if self.vocabulary is None:
            self.vocabulary = set(byte for token in new_sequence for byte in token)
        while iteration < self.iterations:
            counts = {}
            for word in new_sequence:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts: break
            maximum = max(counts, key=counts.get)
            if counts[maximum] == 1: break
            for index, word in enumerate(new_sequence):
                new_word = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == maximum:
                        new_word.append(maximum[0] + maximum[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_sequence[index] = new_word
            self.merges.append(maximum)
            self.vocabulary.add(maximum[0] + maximum[1])
            iteration += 1
        return self.vocabulary, self.merges, new_sequence

    def tokenize(self, sequence):
        return bpe_tokenize(sequence, self.merges, self.space)