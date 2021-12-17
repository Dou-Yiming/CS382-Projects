# coding=utf8
import ipdb
from datasets.dataset import SLUTaggingTrainset
from models.slu_baseline_tagging import SLUTagging
from utils.vocab import PAD
from utils.batch import from_example_list
from utils.example import Example
from utils.initialization import *
from utils.args import init_args
import sys
import os
import time
import gc
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

install_path = os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >=
      0 else "Use CPU as target torch device")

# load data
start_time = time.time()
train_dataset = SLUTaggingTrainset(args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, collate_fn=train_dataset.collate_fn)

args.vocab_size = train_dataset.word_vocab.vocab_size
args.pad_idx = train_dataset.word_vocab[PAD]
args.num_tags = train_dataset.label_vocab.num_tags
args.tag_pad_idx = train_dataset.label_vocab.convert_tag_to_idx(PAD)

# build model
model = SLUTagging(args).to(device)
train_dataset.word2vec.load_embeddings(
    model.word_embed, train_dataset.word_vocab, device=device)

params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
grouped_params = [{'params': list(set([p for n, p in params]))}]
optimizer = AdamW(grouped_params, lr=args.lr)
model.train()
for epoch in range(args.max_epoch):
    epoch_loss = 0.0
    cnt = 0
    for i, batch in tqdm(enumerate(train_loader), leave=False):
        output, loss = model(batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cnt += 1
    print("epoch: {}, loss: {:.4f}".format(epoch, epoch_loss/cnt))


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(
                args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(
                Example.label_vocab, current_batch)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count
