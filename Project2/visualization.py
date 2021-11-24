import numpy as np
import json
import matplotlib.pyplot as plt

from word2vec import CBOW
from utils.vocab import Vocab

model = CBOW.load_model("ckpt")
vocab = Vocab(corpus="./data/treebank.txt", max_vocab_size=-1)
visualize_words = [
    'camera', 'photography',
    'simple', 'easy',
    'child', 'kid',
    'movie', 'film',
    'sea', 'water',
    'eye', 'face',
    'wonderful', 'terrific',
    'music', 'piano',
    'certain', 'sure',
]

visualize_idx = [vocab.token_to_idx(word) for word in visualize_words]
word_vecs = np.array([model.W1[idx] for idx in visualize_idx])
temp = (word_vecs - np.mean(word_vecs, axis=0))
covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualize_words)):
    plt.text(coord[i, 0], coord[i, 1], visualize_words[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig('word_vectors.png')
