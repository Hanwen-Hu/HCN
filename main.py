import argparse
import torch

import IR_Net_Plus
import IR_GAIN_Plus

l_seq_dict = {'PM25':96, 'Traffic':144, 'Solar':144, 'Activity':120}
d_in_dict = {'PM25':36, 'Traffic':214, 'Solar':137, 'Activity':3}

parser = argparse.ArgumentParser()
# Settings of Dataset
parser.add_argument('-dataset', type=str, default='Traffic')
parser.add_argument('-r_miss', type=float, default=0.4, help='Missing Rate')
# Settings of Training
parser.add_argument('-cuda_id', type=int, default=0, help='Cuda ID')
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-n_batch', type=int, default=64)
parser.add_argument('-lr', type=float, default=3e-3, help='Learning Rate')
# Settings of IR-XXX-Plus
parser.add_argument('-plus',type=bool, default=True, help='Usage of Pattern Representation Layer')
parser.add_argument('-iter_time', type=int, default=2, help='Reconstruction Times')
args = parser.parse_args()
args.length = l_seq_dict[args.dataset]
args.dim = d_in_dict[args.dataset]

if torch.cuda.is_available():
    args.device = torch.device('cuda', args.cuda_id)
else:
    args.device = torch.device('cpu')

network = IR_Net_Plus.EXE(args)
network.run()