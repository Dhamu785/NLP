import unicodedata
import time
import glob
import os
from io import open
import unicodedata
import string

import torch as t
from torch.utils.data import Dataset

class helper:
    def __init__(self) -> None:
        self.allowed_char = string.ascii_letters + ".,;'" + "_"
        self.len_allowed_char = len(self.allowed_char)

    def unicode_to_ascii(self, txt: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn' and c in self.allowed_char)


class NameDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        self.data = []
        self.data_tensor = []
        self.label = []
        self.label_tensor = []

        text_files = glob.glob(os.path.join(self.data_dir, '*.txt'))
        for i in text_files:
            label = os.path.splitext(os.path.basename(i))[0]
            labels_set.add(label)
            lines = open(i, 'r', encoding='utf-8').read().strip().split('\n')
            for line in lines:
                self.data.append(line)
                self.label.append(label)
                self.data_tensor.append()