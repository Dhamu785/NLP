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

    def encode_text(self, txt: str) -> t.Tensor:
        name = self.unicode_to_ascii(txt)
        tensor = t.zeros(len(name), 1, self.len_allowed_char)
        for idx, l in enumerate(name):
            if l in self.allowed_char:
                l_idx = self.allowed_char.find(l)
            else:
                l_idx = self.allowed_char.find('_')
            tensor[idx][0][l_idx] = 1
        return tensor

get_tensor = helper()
class NameDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        self.data = []
        self.labels = []
        self.data_tensor = []
        self.label_tensor = []

        text_files = glob.glob(os.path.join(self.data_dir, '*.txt'))
        for i in text_files:
            label = os.path.splitext(os.path.basename(i))[0]
            labels_set.add(label)
            lines = open(i, 'r', encoding='utf-8').read().strip().split('\n')
            for line in lines:
                self.data.append(line)
                self.labels.append(label)
                self.data_tensor.append(get_tensor.encode_text(line))
        
        # Get the labels
        self.unique_labels = list(labels_set)
        for i in self.labels:
            temp = t.tensor([self.unique_labels.index(i)], dtype=t.long)
            self.label_tensor.append(temp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        name = self.data[idx]
        lbl = self.labels[idx]
        name_tensor = self.data_tensor[idx]
        lbl_tensor = self.label_tensor[idx]

        return name, lbl, name_tensor, lbl_tensor