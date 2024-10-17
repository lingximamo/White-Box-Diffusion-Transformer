import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import args


dataset_path = args.dataset_path

dataset = np.load(dataset_path, allow_pickle=True).astype(np.float32)


class CellDataset(Dataset):

    def __init__(self, data, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.data = data

    def __getitem__(self, index):
        return self.data[index][None, :]

    def __len__(self):
        return len(self.data)

    def __load_data__(self, csv_paths: list):
        pass

    def preprocess(self, data):
        pass


cell_dataset = CellDataset(data=torch.tensor(dataset))

bs = args.batch_size

cell_dataloader = DataLoader(dataset=cell_dataset, batch_size=bs, shuffle=True)


