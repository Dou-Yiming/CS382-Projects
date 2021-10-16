from re import L
from lib.model import n_gram_model
from lib.corpus import Corpus
import sys
from tqdm import tqdm

sys.path.append("..")

"""
Implements methods of discounting (count->prob)
Currently support:
1. Good-Turing Smoothing
2. Add-alpha Smoothing
"""
divide_mapping = {
    'quagram': 'trigram',
    'trigram': 'bigram',
    'bigram': 'unigram',
}
num2name = {
    1: 'unigram',
    2: 'bigram',
    3: 'trigram',
    4: 'quagram'
}


def GoodTuringSmoothing(model: n_gram_model, corpus: Corpus):
    # count->seq
    print("Executing Good-Turing Smoothing")
    cnt2seq = {}
    for i in range(2, 5):
        model_name=num2name[i]
        cnt2seq[i] = {}
        seq2cnt = model.model[model_name]
        for v in seq2cnt.values():
            if not v in cnt2seq[i].keys():
                cnt2seq[i][v] = 0
            cnt2seq[i][v] += 1
        l = len(list(seq2cnt.keys())[0])
        cnt2seq[i][0] = pow(corpus.vocab_size, l)-sum(cnt2seq[i].values())
    for i in range(2, 5):
        model_name=num2name[i]
        for k in model.model[model_name].keys():
            if model.model[model_name][k] > 50:
                continue
            model.model[model_name][k] = (model.model[model_name][k] + 1) *\
                cnt2seq[i][model.model[model_name][k]+1] / cnt2seq[i][model.model[model_name][k]]
        # model.model[model_name][tuple(['unseen'])] = cnt2seq[i][1] / cnt2seq[i][0]


def Add_kSmoothing(model: n_gram_model, corpus: Corpus, k=1):
    print("Executing Add-k Smoothing")
    word_sum = sum(model.model['unigram'].values())
    model.model['unigram'] = {word: (cnt + k) / (word_sum + k * corpus.vocab_size)
                              for word, cnt in model.model['unigram'].items()}


def discounting(model: n_gram_model, corpus: Corpus, cfg):
    if cfg.discounting == 'Good-Turing':
        GoodTuringSmoothing(model, corpus)
    elif cfg.discounting == 'Add-k':
        Add_kSmoothing(model, corpus, cfg.k)
    elif cfg.discounting == 'None':
        pass
    else:
        print("ERROR: Unsupported discounting method: {}".format(cfg.discounting))
        exit(1)
