import numpy as np
import os
import os.path as osp
from pprint import pprint


class Corpus:
    def __init__(self, data_path) -> None:
        # dataset: 3 sentences
        print("Loading dataset from {}".format(data_path))
        with open(osp.join(data_path, 'train_set.txt'), 'r') as f:
            self.train_set = f.readlines()[0]
        with open(osp.join(data_path, 'dev_set.txt'), 'r') as f:
            self.dev_set = f.readlines()[0]
        with open(osp.join(data_path, 'test_set.txt'), 'r') as f:
            self.test_set = f.readlines()[0]
        # sentence->list
        print("Building Corpus")
        self.train_set = self.train_set.split(' ')+['</s>']
        self.dev_set = self.dev_set.split(' ')+['</s>']
        self.test_set = self.test_set.split(' ')+['</s>']
        self.db = self.train_set+self.dev_set+self.test_set  # all data
        # vocabulary
        self.vocab_len = {}
        self.vocab_len['uni'] = len(set(self.db))
        self.vocab_len['bi'] = len(set([
            (self.db[i-1], self.db[i]) for i in range(1, len(self.db))
        ]))
        self.vocab_len['tri'] = len(set([
            (self.db[i-2], self.db[i-1], self.db[i]) for i in range(2, len(self.db))
        ]))
        self.vocab_len['qua'] = len(set([
            (self.db[i-3], self.db[i-2], self.db[i-1], self.db[i]) for i in range(3, len(self.db))
        ]))
        pprint(self.vocab_len)