from re import L
from lib.corpus import Corpus
from lib.model import n_gram_model
from lib.discounting import discounting
import sys
from tqdm import tqdm

divide_mapping = {
    'quagram': 'trigram',
    'trigram': 'bigram',
    'bigram': 'unigram',
}


def init_model(model: n_gram_model, corpus: Corpus):
    # init every probability
    print("Initializing unigram model")
    for word in tqdm(corpus.vocab):
        model.model['unigram'][tuple(word)] = 0
    print("Initializing bigram model")
    for i in tqdm(range(corpus.vocab_size)):
        for j in range(corpus.vocab_size):
            key = (corpus.vocab[i], corpus.vocab[j])
            model.model['bigram'][key] = 0


def count_phrase(model: n_gram_model, train_set):
    for phrase, tuples in train_set.items():
        print("Counting {} phrases".format(phrase))
        for key in tqdm(tuples):
            if not key in model.model[phrase].keys():
                model.model[phrase][key] = 0
            model.model[phrase][key] += 1


def cnt2prob(model: n_gram_model):
    print("Computing probability")
    model.model['quagram'] = {k: v/model.model['trigram'][(k[0], k[1], k[2])]
                              for k, v in model.model['quagram'].items()}
    model.model['trigram'] = {k: v/model.model['bigram'][(k[0], k[1])]
                              for k, v in model.model['trigram'].items()}
    model.model['bigram'] = {k: v/model.model['unigram'][tuple([k[0]])]
                             for k, v in model.model['bigram'].items()}
    word_sum = sum(model.model['unigram'].values())
    model.model['unigram'] = {k: v/word_sum for k,
                              v in model.model['unigram'].items()}


def train(model: n_gram_model, corpus: Corpus,cfg):
    # init_model(model,corpus)
    print("Training n-gram Models")
    count_phrase(model, corpus.train_set)
    discounting(model,corpus,cfg)
    cnt2prob(model)
    print("Model size: {}".format(
        {name: len(v.keys()) for name, v in model.model.items()}
    ))
