import pandas as pd
import torch
from torch.utils.data import Dataset

torch.manual_seed(100)
torch.cuda.manual_seed(100)

class TimeSeries(Dataset):
    def __init__(self, l_window, l_step, r_miss):
        self.l_window = l_window  # 每条序列的长度
        self.l_step = l_step  # 每条序列间的距离
        self.r_miss = r_miss
        self.data = None  # 需要修补的残缺序列
        self.label = None  # 完整序列
        self.mask = None  # 需要修补的位置，0为仅data缺失，1为观测数据

    def __getitem__(self, item):
        x = self.data[item * self.l_step : item * self.l_step + self.l_window]
        y = self.label[item * self.l_step : item * self.l_step + self.l_window]
        m = self.mask[item * self.l_step : item * self.l_step + self.l_window]
        return x, y, m

    def __len__(self):
        return (self.data.shape[0] - self.l_window) // self.l_step + 1

    # 标准化数据
    def standardize(self):
        # 若label中存在残缺数据，则用均值填充它
        avg = torch.nanmean(self.label, dim=0, keepdim=True).repeat(self.label.shape[0], 1)
        self.data[torch.isnan(self.label)] = avg[torch.isnan(self.label)]
        self.label[torch.isnan(self.label)] = avg[torch.isnan(self.label)]
        scale = torch.std_mean(self.label, dim=0, keepdim=True)
        self.label = (self.label - scale[1]) / (scale[0])
        self.data = (self.data - scale[1]) / (scale[0])

    def mask_data(self):
        p = torch.rand(self.label.shape)
        self.data[p<self.r_miss] = torch.nan
        self.mask[p<self.r_miss] = 0


class AirQuality(TimeSeries):
    def __init__(self, l_window, l_step, device, r_miss=0.2):
        super().__init__(l_window, l_step, r_miss)

        # 读取数据，PM2.5数据自身有残缺的数据和对应的标签，因此可以直接读取
        data = pd.read_csv('Dataset/PM25/pm25_missing.txt', delimiter=',')
        label = pd.read_csv('Dataset/PM25/pm25_ground.txt', delimiter=',')
        self.data = torch.Tensor(data.iloc[:, 1:].values).to(device)
        self.label = torch.Tensor(label.iloc[:, 1:].values).to(device)
        self.standardize()

        # 由于不需要生成数据，因此依据残缺数据即可生成mask
        self.mask = torch.ones_like(self.data, device=device)
        self.mask[torch.isnan(self.data)] = 0
        self.mask_data()


class Traffic(TimeSeries):
    def __init__(self, l_window, l_step, device, r_miss=0.2):
        super().__init__(l_window, l_step, r_miss)
        data = pd.read_csv('Dataset/traffic.csv', delimiter=',')
        self.data = torch.Tensor(data.iloc[:, 3:].values).to(device)
        self.label = torch.Tensor(data.iloc[:, 3:].values).to(device)
        self.standardize()
        self.mask = torch.ones_like(self.data, device=device)
        self.mask[torch.isnan(self.data)] = 0
        self.mask_data()


class Solar(TimeSeries):
    def __init__(self, l_window, l_step, device, r_miss=0.2):
        super().__init__(l_window, l_step, r_miss)
        data = pd.read_csv('Dataset/solar_AL.csv', delimiter=',')
        self.data = torch.Tensor(data.values).to(device)
        self.label = torch.Tensor(data.values).to(device)
        self.standardize()
        self.mask = torch.ones_like(self.data, device=device)
        self.mask[torch.isnan(self.data)] = 0
        self.mask_data()


class Activity(TimeSeries):
    def __init__(self, l_window, l_step, device, r_miss=0.2):
        super().__init__(l_window, l_step, r_miss)
        data = pd.read_csv('Dataset/activity.csv', delimiter=',')
        self.data = torch.Tensor(data.iloc[:, 4:7].values).to(device)
        self.label = torch.Tensor(data.iloc[:, 4:7].values).to(device)

        self.standardize()
        self.mask = torch.ones_like(self.data, device=device)
        self.mask[torch.isnan(self.data)] = 0
        self.mask_data()
