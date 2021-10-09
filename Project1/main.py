import sys
from tqdm import tqdm
import argparse

sys.path.append("..")

from tools.test import test
from tools.train import train
from tools.test import test
from lib.discounting import discounting
from lib.model import n_gram_model
from lib.corpus import Corpus

def parse_args():
    parser = argparse.ArgumentParser(description='n-gram language model')
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='path of dataset',
                        default='../data/', type=str)
    parser.add_argument('--discounting', dest='discounting',
                        default='Good-Turing', type=str)
    parser.add_argument('-s', dest='save_model',
                        default=False, type=bool)
    parser.add_argument('--alpha', dest='alpha',
                        default=1, type=float)
    args = parser.parse_args()
    return args

def main(args):
    model = n_gram_model()
    corpus = Corpus(args.dataset_path)
    train(model, corpus)
    discounting(model,corpus,args)
    test(model, corpus.test_set)
    if args.save_model:
        model.save()


if __name__ == '__main__':
    args = parse_args()
    main(args)