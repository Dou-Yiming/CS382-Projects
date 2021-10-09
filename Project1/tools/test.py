import sys
from tqdm import tqdm
import numpy as np
import os.path as osp
import json

sys.path.append("..")

from lib.corpus import Corpus
from lib.model import n_gram_model

def log_p(model: n_gram_model,seq):
    assert 1 <= len(seq) and len(seq) <= 4

def save(res,res_path='../result/'):
    print("Saving results")
    with open(osp.join(res_path,'PPL.json'),'w') as f:
        json.dump(res,f)

def test(model: n_gram_model, test_set: list):
    # computes the PPL of test_set
    res={}
    for i in range(1,5):
        print("Testing Perplexity with {}-gram model".format(i))
        ans=0
        if i == 1:
            for j in tqdm(range(0, len(test_set))):
                seq = (test_set[j])
                ans += np.log2(model.unigram[seq])
            ans = pow(2, -1/len(test_set)*ans)
            res['unigram']=ans
        elif i == 2:
            for j in tqdm(range(1, len(test_set))):
                seq = (test_set[i-1],test_set[i])
                ans += np.log2(model.bigram[seq])
            ans = pow(2, -1/len(test_set)*ans)
            res['bigram']=ans
        elif i == 3:
            for j in tqdm(range(2, len(test_set))):
                seq = (test_set[i-2],test_set[i-1],test_set[i])
                ans += np.log2(model.trigram[seq])
            ans = pow(2, -1/len(test_set)*ans)
            res['trigram']=ans
        elif i == 4:
            for j in tqdm(range(3, len(test_set))):
                seq = (test_set[i-3],test_set[i-2],test_set[i-1],test_set[i])
                ans += np.log2(model.quagram[seq])
            ans = pow(2, -1/len(test_set)*ans)
            res['quagram']=ans
        print("PPL = {}".format(ans))
    save(res)