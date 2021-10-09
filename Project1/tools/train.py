import sys
from tqdm import tqdm
sys.path.append("..")
from lib.model import n_gram_model
from lib.corpus import Corpus

def train(model: n_gram_model, corpus: Corpus):
    train_set = corpus.train_set
    db = corpus.db
    # init
    for word in db:
        model.unigram[word] = 0
    for i in range(1, len(db)):
        key = (db[i-1], db[i])
        model.bigram[key] = 0
    for i in range(2, len(db)):
        key = (db[i-2], db[i-1], db[i])
        model.trigram[key] = 0
    for i in range(3, len(db)):
        key = (db[i-3], db[i-2], db[i-1], db[i])
        model.quagram[key] = 0
    # train unigram
    print("Training unigram")
    for word in tqdm(train_set):
        model.unigram[(word)] += 1
    # train bigram
    print("Training bigram")
    for i in tqdm(range(1, len(train_set))):
        key = (train_set[i-1], train_set[i])
        model.bigram[key] += 1
    # # train trigram
    print("Training trigram")
    for i in tqdm(range(2, len(train_set))):
        key = (train_set[i-2], train_set[i-1], train_set[i])
        model.trigram[key] += 1
    # # train quagram
    print("Training quagram")
    for i in tqdm(range(3, len(train_set))):
        key = (train_set[i-3], train_set[i-2], train_set[i-1], train_set[i])
        model.quagram[key] += 1