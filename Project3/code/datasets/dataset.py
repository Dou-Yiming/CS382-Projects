import sys
import os
import time
import gc
import torch
from torch.utils.data import Dataset, DataLoader


class SLUTaggingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
