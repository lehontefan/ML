import nltk
import numpy as np
from nltk.corpus import brown

from mylib.BPE import BytePairEncoder
from mylib.Skipgram import Skipgram
from mylib.Transformer import Transformer

nltk.download('brown')
corpus_text = ' '.join(brown.words()[:100000])

bpe = BytePairEncoder(iterations=10000)
vocabulary, merges, tokenized_words = bpe.fit(corpus_text)

token_sequence = [token for word in tokenized_words for token in word]

print(f"Размер словаря BPE: {len(vocabulary)}")
print(f"Длина токенизированной последовательности: {len(token_sequence)}")

d_model = 64
skipgram = Skipgram(size=d_model, window=4, learning_rate=0.01)
skipgram.fit(token_sequence, n_epochs=100)

vocab_list = sorted(vocabulary)
vocab_size = len(vocab_list)
token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
id_to_token = {i: tok for tok, i in token_to_id.items()}

def token_to_embedding(token):
    return skipgram.w_target.get(token, np.zeros(d_model))

seq_len = 16
batch_size = 8

def make_batch(token_sequence, seq_len, batch_size, start_idx):
    X_batch = np.zeros((batch_size, seq_len, d_model))
    y_batch = np.zeros((batch_size, seq_len, vocab_size))
    for b in range(batch_size):
        offset = start_idx + b * seq_len
        window = token_sequence[offset: offset + seq_len]
        target = token_sequence[offset + 1: offset + seq_len + 1]
        for t, tok in enumerate(window):
            X_batch[b, t, :] = token_to_embedding(tok)
        for t, tok in enumerate(target):
            y_batch[b, t, token_to_id[tok]] = 1
    return X_batch, y_batch

X_train, y_train = make_batch(token_sequence, seq_len, batch_size, start_idx=0)

transformer = Transformer(
    blocks=2,
    heads=4,
    d_model=d_model,
    d_key=16,
    d_value=16,
    max_seq_len=seq_len,
    vocab_size=vocab_size,
    iterations=10000,
    learning_rate=0.01
)
transformer.fit(X_train, y_train)

def predict_next(context_tokens):
    X = np.zeros((1, len(context_tokens), d_model))
    for t, tok in enumerate(context_tokens):
        X[0, t, :] = token_to_embedding(tok)
    probs = transformer.predict(X)
    next_probs = probs[0, -1, :]
    next_id = np.argmax(next_probs)
    return id_to_token[next_id]

for i in range(10):
    context = token_sequence[i*16:seq_len+i*16]
    next_token = predict_next(context)
    print(f"Следующий токен: {next_token}", " - ", token_sequence[seq_len+i*16])