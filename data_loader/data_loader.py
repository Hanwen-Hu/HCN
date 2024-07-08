import torch
from torch.utils.data import Dataset
import numpy as np

torch.manual_seed(980525)
torch.cuda.manual_seed(980525)


class TSDataset(Dataset):
    def __init__(self, dataset, length, device, miss_rate, step=1):
        self.length = length
        self.step = step
        path1 = 'dataset/' + dataset + '_' + str(int(miss_rate * 10)) + '.txt'
        path2 = 'dataset/' + dataset + '.txt'
        self.data = torch.Tensor(np.genfromtxt(path1, dtype=float, delimiter=',')).to(device)
        self.label = torch.Tensor(np.genfromtxt(path2, dtype=float, delimiter=',')).to(device)
        self._preprocess()
        self._mask()

    def _preprocess(self):
        avg = torch.nanmean(self.label, dim=0, keepdim=True).repeat(self.label.shape[0], 1)
        temp_data = self.label.clone()
        temp_data[torch.isnan(self.label)] = avg[torch.isnan(self.label)]
        std = torch.std(temp_data, dim=0, keepdim=True)
        self.label = (self.label - avg) / std
        self.data = (self.data - avg) / std

    def _mask(self):
        self.data_mask = torch.ones_like(self.data)
        self.data_mask[torch.isnan(self.data)] = 0  # 自然缺失 + 主动缺失
        self.test_mask = self.data_mask.clone()
        self.test_mask[torch.isnan(self.label)] = 1  # 主动缺失

    def __len__(self):
        return (self.label.shape[0] - self.length) // self.step + 1


class InSampleDataset(TSDataset):
    def __init__(self, dataset, length, device, miss_rate, step=1):
        super().__init__(dataset, length, device, miss_rate, step)
        self._valid_mask()

    def _valid_mask(self):
        self.valid_mask = torch.rand_like(self.data)
        self.valid_mask[self.valid_mask >= 0.1] = 1
        self.valid_mask[self.valid_mask < 0.1] = 0
        self.valid_mask[self.data_mask == 0] = 1

    def __getitem__(self, index):
        left = index * self.step
        right = left + self.length
        return self.data[left: right], self.label[left: right], self.data_mask[left: right], self.test_mask[left: right], self.valid_mask[left: right]


class OutOfSampleDataset(TSDataset):
    def __init__(self, dataset, length, device, miss_rate, step=1, mode='train'):
        super().__init__(dataset, length, device, miss_rate, step)
        self._split(mode)

    def _split(self, mode='train'):
        if mode == 'train':
            left, right = 0, int(self.data.shape[0] * 0.6)
        elif mode == 'valid':
            left, right = int(self.data.shape[0] * 0.6), int(self.data.shape[0] * 0.7)
        else:
            left, right = int(self.data.shape[0] * 0.7), self.data.shape[0]
        self.data = self.data[left:right]
        self.label = self.label[left:right]
        self.data_mask = self.data_mask[left:right]
        self.test_mask = self.test_mask[left:right]

    def __getitem__(self, index):
        left = index * self.step
        right = left + self.length
        return self.data[left: right], self.label[left: right], self.data_mask[left: right], self.test_mask[left: right]
