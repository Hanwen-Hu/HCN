import torch
from torch.utils.data import DataLoader

from Loader import AirQuality, Traffic, Solar, Activity
from IR_GAIN_Plus.structure import Generator, Discriminator

class EXE:
    def __init__(self, args):
        self.args = args
        self.dict = {'PM25': AirQuality, 'Traffic': Traffic, 'Solar':Solar, 'Activity':Activity}
        self.generator = Generator(args.length, args.length, args.device, args.plus).to(args.device)
        self.discriminator = Discriminator(args.length, args.length).to(args.device)
        self.dataset = self.dict[args.dataset](args.length, 1, args.device, args.miss_rate)
        self.criterion = torch.nn.MSELoss()

    def run(self):
        mse_loss = torch.nn.MSELoss()
        mae_loss = lambda a, b: torch.mean(torch.abs(a - b))
        optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=self.args.lr)
        optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.lr)
        loader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        for epoch in range(self.args.epochs * 10):
            mse_error, mae_error, batch_num = 0, 0, 0
            for i, (x, y, m) in enumerate(loader):
                # Training
                x[m==0] = 0
                optimizer_g.zero_grad()
                x_1 = self.generator(x, m)
                x_1_ = x_1.clone()
                x_1_[m==1] = x[m==1]
                p = self.discriminator(x_1_, m)
                loss_g = 3 * mse_loss(out[m==1], x[m==1]) - mse_loss(p, m)
                x_tilde = self.generator(out, 1 - m)
                loss_g = loss_g + 2 * mse_loss(x_tilde[m==1], x[m==1])
                loss_g.backward()
                optimizer_g.step()

                # Testing
                optimizer_d.zero_grad()
                out = self.generator(x, m)
                out[m==1] = x[m==1]
                p = self.discriminator(out, m)
                loss_d = mse_loss(p, m)
                loss_d.backward()
                optimizer_d.step()

                mse_error += mse_loss(out[m==0], y[m==0]).item()
                mae_error += mae_loss(out[m==0], y[m==0]).item()
                batch_num += 1
            print(epoch, round(mse_error/ batch_num, 4), round(mae_error / batch_num, 4))
            torch.save(self.generator, 'Files/IRGAINPlus_' + self.args.dataset + '_' + str(int(self.args.r_miss * 10)) + '.pth')