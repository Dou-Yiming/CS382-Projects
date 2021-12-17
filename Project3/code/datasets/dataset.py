import json
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.vocab import Vocab, LabelVocab, PAD
from utils.word2vec import Word2vecUtils


class SLUTaggingTrainset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_len = args.max_len
        self.max_tag_len = args.max_tag_len
        # load raw data
        self.data_path = osp.join(args.dataroot, 'train.json')
        self.json_data = json.load(open(self.data_path, 'r'))
        # configuration
        self.word_vocab = Vocab(padding=True, unk=True,
                                filepath=self.data_path)
        self.word2vec = Word2vecUtils(args.word2vec_path)
        self.label_vocab = LabelVocab(args.dataroot)
        # data preprocessing
        self.data = []
        self.load_data()

    def load_data(self):
        for d in self.json_data:
            for ex in d:
                self.data.append(self.utt2slot(ex))

    def utt2slot(self, ex):
        # utt, slot, tags, slotvalue, input_idx, tag_id
        ans = {}
        ans['utt'] = ex['asr_1best']
        ans['slot'] = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                ans['slot'][act_slot] = label[2]
        ans['tags'] = ['O']*len(ans['utt'])
        for slot, value in ans['slot'].items():
            bidx = ans['utt'].find(value)
            if bidx == -1:
                continue
            ans['tags'][bidx:bidx+len(value)] = [f'I-{slot}'] * len(value)
            ans['tags'][bidx] = f'B-{slot}'
        ans['slotvalue'] = [f'{slot}-{value}' for slot,
                            value in ans['slot'].items()]
        ans['input_idx'] = [self.word_vocab[c] for c in ans['utt']]
        ans['tag_id'] = [self.label_vocab.convert_tag_to_idx(
            tag)for tag in ans['tags']]
        return ans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pad_idx = self.args.pad_idx
        tag_pad_idx = self.args.tag_pad_idx
        # utt
        data = self.data[index]
        utt, input_idx = data['utt'], data['input_idx']
        length = len(input_idx)
        # slotvalue
        slotvalue = data['slotvalue']
        # input_ids
        input_ids = input_idx+[pad_idx]*(self.max_len-len(input_idx))
        # tag_id, tag_mask
        tag_id = data['tag_id']
        tag_mask = [1] * len(tag_id) + [0] * (self.max_tag_len - len(tag_id))
        tag_id = tag_id + [tag_pad_idx] * (self.max_tag_len - len(tag_id))
        input_ids = np.array(input_ids)
        tag_id = np.array(tag_id)
        tag_mask = np.array(tag_mask)
        return [input_ids, tag_id, tag_mask, length, utt, slotvalue]

    def collate_fn(self, data):
        blob = list(zip(*data))
        output = {}
        output['input_ids'] = torch.tensor(np.stack(blob[0]),
                                           dtype=torch.long,
                                           device=self.args.device)
        output['tag_ids'] = torch.tensor(np.stack(blob[1]),
                                        dtype=torch.long,
                                        device=self.args.device)
        output['tag_mask'] = torch.tensor(np.stack(blob[2]),
                                          dtype=torch.float,
                                          device=self.args.device)
        output['lengths'] = torch.tensor(blob[3], dtype=torch.long)
        output['utt'] = blob[4]
        output['slotvalue'] = blob[5]
        return output
