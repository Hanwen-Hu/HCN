import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Loader import Activity, AirQuality, Traffic, Solar
from IR_Square_Net.structure import Net


class EXE:
    def __init__(self, args, load=False):
        self.args = args
        self.dict = {'PM25': AirQuality, 'Traffic': Traffic, 'Solar': Solar, 'Activity': Activity}
        self.dataset = self.dict[args.dataset](args.length, 1, args.device, args.r_miss)
        if load:
            self.model = torch.load('Files/IR-Square-Net_' + args.dataset + '_' + str(int(args.miss_rate * 10)) + '.pth')
        else:
            self.model = Net(args.dim, args.length, args.device, args.irm_usage).to(args.device)
        self.criterion = nn.MSELoss()

    def iterative_reconstruction(self, x, m, iter_time=2):
        x[m == 0] = 0
        loss = 0
        iter_x, iter_m = x, m
        for _ in range(iter_time):
            iter_x = self.model(iter_x, iter_m)
            loss = loss + self.criterion(iter_x[m == 1], x[m == 1])
            iter_m = 1 - iter_m
        return loss

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        loader = DataLoader(self.dataset, batch_size=self.args.n_batch, shuffle=True)
        mae_loss = lambda a, b: torch.mean(torch.abs(a - b))
        for i in range(self.args.epochs):
            print('Epoch', i)
            n_batch, mse, mae, loss_train = 0, 0, 0, 0
            for _, (x, y, m) in enumerate(loader):
                # Training
                self.model.train()
                optimizer.zero_grad()
                loss = self.iterative_reconstruction(x, m, self.args.iter_time)
                loss_train += loss.item()
                loss.backward()
                optimizer.step()
                n_batch += 1

                # Testing
                self.model.eval()
                x[m == 1] = y[m == 1]
                x_full = self.model(x, m)
                mse += self.criterion(x_full[m == 0], y[m == 0]).item()
                mae += mae_loss(x_full[m == 0], y[m == 0]).item()

            print('MSE', round(mse / n_batch, 4))
            print('MAE', round(mae / n_batch, 4))
            torch.save(self.model, 'Files/IR-Square-Net_' + self.args.dataset + '_' + str(int(self.args.r_miss * 10)) + '.pth')
