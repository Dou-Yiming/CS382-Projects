from lib.model import n_gram_model
from lib.corpus import Corpus
import sys
from tqdm import tqdm
import numpy as np
import os.path as osp
import json

num2name = {
    1: 'unigram',
    2: 'bigram',
    3: 'trigram',
    4: 'quagram'
}


def save(PPL, cfg, res_path='./result/result.json'):
    print("Saving results to {}".format(res_path))
    res = dict(cfg)
    res['PPL'] = PPL
    with open(res_path, 'w') as f:
        json.dump(res, f)


def replace_UNK(model: n_gram_model, corpus: Corpus):
    # replace the unseen word in test set with <UNK>
    for i in range(1, len(corpus.test_set)-1):
        key = tuple([corpus.test_set[i]])
        if not key in model.model['unigram'].keys():
            corpus.test_set[i] = '<UNK>'
    corpus.test_set = corpus.sentence2phrase(corpus.test_set)


def prob(model: n_gram_model, key, lambd):
    if len(key) == 1:
        return model.model['unigram'][key]

    model_name = num2name[len(key)]
    if not key in model.model[model_name].keys():
        return (1 - lambd) * prob(model, tuple(key[1:]), lambd)
    return lambd * model.model[model_name][key] +\
        (1-lambd) * prob(model, tuple(key[1:]), lambd)


def test(model: n_gram_model, corpus: Corpus, cfg):
    replace_UNK(model, corpus)
    lams=[0.05,0.1,0.2,0.4,0.8,0.95]
    for l in lams:
        cfg.lambd=l
        print("Testing (lambda = {})".format(cfg.lambd))
        PPL = 0
        for key in tqdm(corpus.test_set[cfg.model]):
            PPL += np.log2(prob(model, key, cfg.lambd))
        PPL = pow(2, -1 / len(corpus.test_set['unigram']) * PPL)
        print("PPL = {:.2f}".format(PPL))
        save(PPL, cfg)
