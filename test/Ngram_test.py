import random
import nltk
from nltk.corpus import brown
import pickle

from mylib.BPE import BytePairEncoder
from mylib.Ngram import Ngram
from mylib.Tokenizer import bpe_tokenize, byte_tokenize

"""
nltk.download('brown')
corpus = ' '.join(brown.words())

bpe = BytePairEncoder()
bpe.fit(corpus)

ngram = Ngram(5, smoothing=0.5)
ngram.fit(corpus, bpe)

with open('bpe.pkl', 'wb') as f:
    pickle.dump(bpe, f)

with open('ngram.pkl', 'wb') as f:
    pickle.dump(ngram, f)
    """

with open('bpe.pkl', 'rb') as f:
    bpe = pickle.load(f)
with open('ngram.pkl', 'rb') as f:
    ngram = pickle.load(f)

def sample_next_token(ngram, sequence, temperature=1.0):
    if ngram.merges is not None:
        new_sequence = bpe_tokenize(sequence, ngram.merges, ngram.space)
    else:
        new_sequence = byte_tokenize(sequence, ngram.space)
    new_sequence = [token for word in new_sequence for token in word]

    probabilities = []
    for token in ngram.vocabulary:
        probability = 0
        for order in range(1, ngram.n + 1):
            if order == 1:
                history = ()
            else:
                history = tuple(new_sequence[-(order - 1):])
            gram = history + (token,)
            num = ngram.counts.get(gram, 0) + ngram.smoothing
            denom = ngram.counts.get(history, 0) + ngram.smoothing * len(ngram.vocabulary)
            p = num / denom if denom > 0 else 0.0
            probability += ngram.lambdas[order - 1] * p
        probabilities.append(probability)

    if temperature != 1.0:
        probabilities = [p ** (1.0 / temperature) for p in probabilities]

    total = sum(probabilities)
    if total == 0:
        return random.choice(ngram.vocabulary)

    weights = [p / total for p in probabilities]
    return random.choices(ngram.vocabulary, weights=weights, k=1)[0]


def sample_text(ngram, start_text, num_tokens=50, temperature=1.0):
    text = start_text
    for _ in range(num_tokens):
        token_hex = sample_next_token(ngram, text, temperature)
        text += bytes.fromhex(token_hex).decode('utf-8', errors='ignore')
    return text

result = sample_text(ngram, 'The Fulton County Grand', num_tokens=500, temperature=1.0)
print(result)