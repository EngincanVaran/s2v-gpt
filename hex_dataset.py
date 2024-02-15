import torch
from torch.utils.data import Dataset

import utils


class HexDataset(Dataset):
    def __init__(self, file_path, block_size=32):
        self.block_size = block_size
        self.data = self.load_and_encode(file_path)

    def load_and_encode(self, file_path):
        with open(file_path, "r") as file:
            data = [line.strip() for line in file.readlines()]
        encoded_data = torch.tensor(utils.encode(data))
        return encoded_data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.block_size], self.data[idx + 1:idx + self.block_size + 1]
