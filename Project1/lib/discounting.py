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


def GoodTuringSmoothing(model: n_gram_model, corpus: Corpus):
    # count->seq
    print("Executing Good-Turing Smoothing")
    cnt2seq = {}
    seq2cnt_list = [model.unigram, model.bigram, model.trigram, model.quagram]
    for i in range(0, 4):
        cnt2seq[i+1] = {}
        seq2cnt = seq2cnt_list[i]
        for v in seq2cnt.values():
            if not v in cnt2seq[i+1].keys():
                cnt2seq[i+1][v] = 0
            cnt2seq[i+1][v] += 1
    for k in model.unigram.keys():
        if model.unigram[k]>50:
            continue
        if model.unigram[k]+1 in cnt2seq[1].keys():
            model.unigram[k] = (model.unigram[k] + 1)*cnt2seq[1][model.unigram[k]+1]/cnt2seq[1][model.unigram[k]]
    for k in model.bigram.keys():
        if model.bigram[k]>50:
            continue
        if model.bigram[k]+1 in cnt2seq[2].keys():
            model.bigram[k] = (model.bigram[k]+1)*cnt2seq[2][model.bigram[k]+1]/cnt2seq[2][model.bigram[k]]
    for k in model.trigram.keys():
        if model.trigram[k]>50:
            continue
        if model.trigram[k]+1 in cnt2seq[3].keys():
            model.trigram[k] = (model.trigram[k]+1)*cnt2seq[3][model.trigram[k]+1]/cnt2seq[3][model.trigram[k]]
    for k in model.quagram.keys():
        if model.quagram[k]>50:
            continue
        if model.quagram[k]+1 in cnt2seq[4].keys():
            model.quagram[k] = (model.quagram[k]+1)*cnt2seq[4][model.quagram[k]+1]/cnt2seq[4][model.quagram[k]]
    model.quagram = {k: v/model.trigram[(k[0], k[1], k[2])]
                     for k, v in model.quagram.items()}
    model.trigram = {k: v/model.bigram[(k[0], k[1])]
                     for k, v in model.trigram.items()}
    model.bigram = {k: v/model.unigram[(k[0])]
                    for k, v in model.bigram.items()}
    word_sum = sum(model.unigram.values())
    model.unigram = {k: v/word_sum for k, v in model.unigram.items()}


def AddAlphaSmoothing(model: n_gram_model, corpus: Corpus, alpha: float):
    print("Executing Add-alpha Smoothing")
    model.quagram = {k: (v+alpha)/(model.trigram[(k[0], k[1], k[2])]+alpha*corpus.vocab_len['qua'])
                     for k, v in model.quagram.items()}
    model.trigram = {k: (v+alpha)/(model.bigram[(k[0], k[1])]+alpha*corpus.vocab_len['tri'])
                     for k, v in model.trigram.items()}
    model.bigram = {k: (v+alpha)/(model.unigram[(k[0])]+alpha*corpus.vocab_len['bi'])
                    for k, v in model.bigram.items()}
    word_sum = sum(model.unigram.values())
    model.unigram = {k: (v+alpha)/(word_sum+alpha*corpus.vocab_len['uni'])
                     for k, v in model.unigram.items()}


def discounting(model: n_gram_model, corpus: Corpus, args):
    if args.discounting == 'Good-Turing':
        GoodTuringSmoothing(model, corpus)
    elif args.discounting == 'AddAlpha':
        AddAlphaSmoothing(model, corpus, args.alpha)
    else:
        print("ERROR: Unsupported discounting method: {}".format(args.discounting))
        exit(1)