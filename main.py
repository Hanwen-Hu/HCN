import argparse
import torch

from model import IR2InSample, IR2OutOfSample
from data_loader import generate_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Model Selection
    parser.add_argument('-model', type=str, default='Net', help='Net or GAN')
    parser.add_argument('-mode', type=str, default='InSample', help='InSample or OutOfSample')
    # Settings of Dataset
    parser.add_argument('-dataset', type=str, default='pm25', help='pm25, activity, traffic, solar')
    parser.add_argument('-r_miss', type=float, default=0.2, help='Missing Rate')
    # Settings of Training
    parser.add_argument('-cuda_id', type=int, default=0, help='Cuda ID')
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-n_batch', type=int, default=64)
    parser.add_argument('-lr', type=float, default=3e-3, help='Learning Rate')
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-load', action='store_true')
    # Settings of Model
    parser.add_argument('-use_irm', type=int, default=1, help='Usage of Incomplete Representation Mechanism')
    parser.add_argument('-iter_time', type=int, default=2, help='Reconstruction Times')
    parser.add_argument('-dropout', type=float, default=0.05, help='Dropout Rate')
    parser.add_argument('-n_layer', type=int, default=1, help='Number of Layers')
    args = parser.parse_args()
    args.length = 96
    args.step = 1
    dim_dict = {'pm25': 36, 'traffic': 214, 'solar': 137, 'activity': 3}
    args.dim = dim_dict[args.dataset]

    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


def main() -> None:
    args = parse_arguments()
    print('Dataset: {}'.format(args.dataset.upper()))
    print('------Dataset Generation-------')
    generate_dataset(args.dataset, args.r_miss)
    print('------Model Configuration------')
    print("IR2{}, {}, Iter Time: {}, Use IRM: {}, Max Epoch: {}".format(args.model, args.mode, args.iter_time, bool(args.use_irm), args.epochs))
    if args.mode == 'InSample':
        model = IR2InSample(args)
    else:
        model = IR2OutOfSample(args)
    print('-----------Training------------')
    model.train()
    print('------------Testing------------')
    model.test()


if __name__ == '__main__':
    main()
