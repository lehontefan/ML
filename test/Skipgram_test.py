import nltk
import pickle
from nltk.corpus import brown

from mylib.Skipgram import Skipgram
"""
nltk.download('brown')
corpus = brown.words()

skipgram = Skipgram(window=4, size=300)
skipgram.fit(corpus)

with open('skipgram.pkl', 'wb') as f:
    pickle.dump(skipgram, f)
"""
with open('skipgram.pkl', 'rb') as f:
    skipgram = pickle.load(f)

print(skipgram.get_similar('number', 5))