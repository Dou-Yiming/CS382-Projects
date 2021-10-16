import numpy as np
import os
import os.path as osp
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt



class Corpus:
    def __init__(self, data_path, cfg) -> None:
        self.data_path = data_path
        self.cfg=cfg
        self.build_corpus()

    def load_data(self):
        print("Loading dataset from {}".format(self.data_path))
        with open(osp.join(self.data_path, 'train_set.txt'), 'r') as f:
            train_sentence = f.readlines()[0]
        with open(osp.join(self.data_path, 'test_set.txt'), 'r') as f:
            test_sentence = f.readlines()[0]
        return ['<s>'] + train_sentence.split(' ') + ['</s>'],\
               ['<s>'] + test_sentence.split(' ') + ['</s>']

    def few2unk(self, train_sentence, thres):
        # count
        word2cnt = {}
        for word in train_sentence:
            if not word in word2cnt.keys():
                word2cnt[word] = 0
            word2cnt[word] += 1
        # plot
        word2cnt_sort = sorted(word2cnt.items(), key=lambda x:x[1], reverse=True)
        plt.plot(range(len(word2cnt_sort)), [np.log10(x[1]) for x in word2cnt_sort])
        plt.title("word2cnt_sort")
        plt.savefig('./visualization/train_set.jpg')
        plt.close()
        # replace
        for i in range(1, len(train_sentence)-1):
            if word2cnt[train_sentence[i]] <= thres:
                train_sentence[i] = '<UNK>'
        return train_sentence

    def sentence2phrase(self,sentence):
        ans = {}
        l = len(sentence)
        ans['unigram'] = [tuple([sentence[i]]) for i in range(0, l)]
        ans['bigram'] = [(sentence[i-1], sentence[i]) for i in range(1, l)]
        ans['trigram'] = [(sentence[i-2], sentence[i-1], sentence[i])
                          for i in range(2, l)]
        ans['quagram'] = [(sentence[i-3], sentence[i-2], sentence[i-1], sentence[i])
                          for i in range(3, l)]
        return ans
    
    def build_corpus(self):
        train_sentence, test_sentence = self.load_data()
        print("Building Corpus")
        train_sentence = self.few2unk(train_sentence,self.cfg.unk_thres)
        self.train_set = self.sentence2phrase(train_sentence)
        self.test_set = test_sentence
        
        self.vocab = list(set(train_sentence))
        self.vocab_size = len(self.vocab)
        
        print("Vocabulary size: {}".format(self.vocab_size))