from lib.corpus import Corpus
from lib.model import n_gram_model
from lib.discounting import discounting
from tools.train import train
from tools.test import test
from ctypes import sizeof
import sys
from tqdm import tqdm
import argparse
import json
from easydict import EasyDict as edict

sys.path.append("..")


def parse_args():
    parser = argparse.ArgumentParser(description='n-gram language model')
    parser.add_argument('--data_path', dest='data_path',
                        help='path of dataset',
                        default='./data/', type=str)
    parser.add_argument('--config_path', dest='config_path',
                        help='path of config',
                        default='./configs/default.json', type=str)
    args = parser.parse_args()
    return args


def get_config(args):
    config_path = args.config_path
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    return cfg


def main(args, cfg):
    model = n_gram_model()
    corpus = Corpus(args.data_path, cfg)
    if cfg.train:
        train(model, corpus,cfg)
        if cfg.save:
            model.save()
    else:
        model.load()
    test(model, corpus, cfg)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(args, cfg)
