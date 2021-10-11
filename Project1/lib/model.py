import json
import os
import os.path as osp
import pickle


class n_gram_model:
    def __init__(self):
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.quagram = {}

    def save(self, model_path='../trained_model/model.pkl'):
        print('Saving model to {}'.format(model_path))
        model = {'unigram': self.unigram,
                 'bigram': self.bigram,
                 'trigram': self.trigram,
                 'quagram': self.quagram}
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load(self, model_path='../trained_model/model.pkl'):
        print("Loading model from {}".format(model_path))
        try:
            f = open(model_path, 'rb')
            model = pickle.load(f)
            self.unigram = model['unigram']
            self.bigram = model['bigram']
            self.trigram = model['trigram']
            self.quagram = model['quagram']
        except:
            print("ERROR: Cannot Load File from {}".format(model_path))