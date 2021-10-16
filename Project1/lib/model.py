import json
import os
import os.path as osp
import pickle

from tqdm.std import tqdm


class n_gram_model:
    def __init__(self):
        self.model = {
            'unigram': {},
            'bigram': {},
            'trigram': {},
            'quagram': {}
        }
        

    def save(self, model_path='./trained_model/model.pkl'):
        print('Saving model to {}'.format(model_path))
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, model_path='./trained_model/model.pkl'):
        print("Loading model from {}".format(model_path))
        try:
            f = open(model_path, 'rb')
            self.model = pickle.load(f)
        except:
            print("ERROR: Cannot Load File from {}".format(model_path))
