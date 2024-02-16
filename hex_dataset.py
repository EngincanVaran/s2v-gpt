import logging
import tarfile

import torch
from torch.utils.data import Dataset

import utils


class HexDataset(Dataset):
    def __init__(self, file_path, block_size=32):
        self.block_size = block_size
        self.data = self.load_and_encode(file_path)

    def load_and_encode(self, file_path):
        tar = tarfile.open(file_path)
        data = []
        for member in tar.getnames():
            if "12bit" in member:
                f = tar.extractfile(member)
                data = f.readlines()
                f = None
                data = [line.rstrip().decode("utf-8") for line in data]
                break
        tar.close()
        encoded_data = torch.tensor(utils.encode(data))
        return encoded_data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.block_size], self.data[idx + 1:idx + self.block_size + 1]
