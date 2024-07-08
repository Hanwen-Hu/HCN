import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import InSampleDataset, OutOfSampleDataset

from .networks import Net, Discriminator


class IR2:
    def __init__(self, args):
        self.args = args
        self.path = 'IR2' + self.args.model + '_' + self.args.dataset + '_' + str(int(self.args.r_miss * 10)) + '.pth'
        self.model = Net(args).to(args.device)
        if args.load:
            state_dict = torch.load(self.path, map_location=args.device)
            self.model = torch.load(state_dict)
        self.discriminator = Discriminator(args).to(args.device) if self.args.model == 'GAN' else None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs)
        self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.args.lr) if self.args.model == 'GAN' else None

        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = lambda a, b: torch.mean(torch.abs(a - b))

    def _get_data(self, mode=None):
        if mode is None:
            dataset = InSampleDataset(self.args.dataset, self.args.length, self.args.device, self.args.r_miss, self.args.step)
        else:
            dataset = OutOfSampleDataset(self.args.dataset, self.args.length, self.args.device, self.args.r_miss, self.args.step, mode)
        return DataLoader(dataset, batch_size=self.args.n_batch, shuffle=True)

    def _iterative_reconstruction(self, x, m, iter_time):
        loss = 0
        iter_x, iter_m = x, m
        for _ in range(iter_time):
            iter_x = self.model(iter_x, iter_m)
            loss = loss + self.mse_loss(iter_x[m == 1], x[m == 1])
            iter_m = 1 - iter_m
        return loss

    def _discriminator_loss(self, x, m):
        x_impute = self.model(x, m)
        x_impute[m == 1] = x[m == 1]
        p = self.discriminator(x_impute, m)
        return self.bce_loss(p, m)

    def _train_batch(self, x, m):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._iterative_reconstruction(x, m, self.args.iter_time)
        if self.args.model == 'GAN':
            loss = loss - self._discriminator_loss(x, m)
        loss.backward()
        self.optimizer.step()
        if self.args.model == 'GAN':
            self.discriminator.train()
            self.optim_d.zero_grad()
            loss = self._discriminator_loss(x, m)
            loss.backward()
            self.optim_d.step()

    def _valid_batch(self, x, m, v):
        self.model.eval()
        x_hat = self.model(x, m)
        return self.mse_loss(x_hat[v == 0], x[v == 0]).item()

    def _patience(self, epoch, valid_loss, best_valid, patience):
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(self.model.state_dict(), self.path)
            patience = 0
        else:
            patience += 1
        print('Epoch', epoch, '\tMSE', round(valid_loss, 4), '\tBest', round(best_valid, 4), '\tPatience', patience)
        return best_valid, patience

    def _test_batch(self, x, y, m, t):
        self.model.eval()
        x_hat = self.model(x, m)
        return self.mse_loss(x_hat[t == 0], y[t == 0]).item(), self.mae_loss(x_hat[t == 0], y[t == 0]).item()

    def _save_result(self, mse, mae, mode):
        print('MSE', round(mse, 4))
        print('MAE', round(mae, 4))
        result = mode + '_' + self.args.model + ',' + self.args.dataset + ',' + str(self.args.r_miss) + ',' + str(self.args.use_irm) + ',' + str(self.args.iter_time) + ',' + str(round(mse, 4)) + ',' + str(round(mae, 4)) + '\n'
        with open('result.txt', 'a') as f:
            f.write(result)


class IR2InSample(IR2):
    def __init__(self, args):
        super().__init__(args)
        self.path = 'networks/In_' + self.path

    def train(self):
        loader = self._get_data()
        patience, best_valid = 0, float('inf')
        for epoch in range(self.args.epochs):
            mse = 0
            for x, _, m, _, v in tqdm(loader):
                mask = m * v
                x[m == 0] = 0
                self._train_batch(x, mask)
                mse += self._valid_batch(x, mask, v)
            mse /= len(loader)
            self.scheduler.step()
            best_valid, patience = self._patience(epoch, mse, best_valid, patience)
            if patience == self.args.patience:
                break

    def test(self):
        state_dict = torch.load(self.path, map_location=self.args.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        loader = self._get_data()
        mse, mae = 0, 0
        with torch.no_grad():
            for x, y, m, t, _ in tqdm(loader):
                x[m == 0] = 0
                error = self._test_batch(x, y, m, t)
                mse += error[0]
                mae += error[1]
        self._save_result(mse / len(loader), mae / len(loader), 'In')


class IR2OutOfSample(IR2):
    def __init__(self, args):
        super().__init__(args)
        self.path = 'networks/Out_' + self.path

    def train(self):
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        patience, best_valid = 0, float('inf')
        for epoch in range(self.args.epochs):
            for x, y, m, t in tqdm(train_loader):
                x[m == 0] = 0
                self._train_batch(x, m)
            self.model.eval()
            mse = 0
            with torch.no_grad():
                for x, y, m, t in tqdm(valid_loader):
                    x[m == 0] = 0
                    mse += self._test_batch(x, y, m, t)[0]
            self.scheduler.step()
            best_valid, patience = self._patience(epoch, mse / len(valid_loader), best_valid, patience)
            if patience == self.args.patience:
                break

    def test(self):
        state_dict = torch.load(self.path, map_location=self.args.device)
        self.model.load_state_dict(state_dict)
        loader = self._get_data('test')
        mse, mae = 0, 0
        with torch.no_grad():
            for x, y, m, t in tqdm(loader):
                x[m == 0] = 0
                error = self._test_batch(x, y, m, t)
                mse += error[0]
                mae += error[1]
        self._save_result(mse / len(loader), mae / len(loader), 'Out')
