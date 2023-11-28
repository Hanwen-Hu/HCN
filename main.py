import argparse
import torch

import IR_Square_Net
import IR_Square_GAIN

model_dict = {'Net':IR_Square_Net, 'GAIN':IR_Square_GAIN}
l_seq_dict = {'PM25':96, 'Traffic':144, 'Solar':144, 'Activity':120}
d_in_dict = {'PM25':36, 'Traffic':214, 'Solar':137, 'Activity':3}

parser = argparse.ArgumentParser()
# Model Selection
parser.add_argument('-model', type=str, default='GAIN', help='Net or GAIN')
# Settings of Dataset
parser.add_argument('-dataset', type=str, default='Traffic')
parser.add_argument('-r_miss', type=float, default=0.2, help='Missing Rate')
# Settings of Training
parser.add_argument('-cuda_id', type=int, default=0, help='Cuda ID')
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-n_batch', type=int, default=64)
parser.add_argument('-lr', type=float, default=3e-3, help='Learning Rate')
# Settings of IR-XXX-Plus
parser.add_argument('-irm_usage',type=bool, default=True, help='Usage of Incomplete Representation Mechanism')
parser.add_argument('-iter_time', type=int, default=2, help='Reconstruction Times')
args = parser.parse_args()
args.length = l_seq_dict[args.dataset]
args.dim = d_in_dict[args.dataset]

if torch.cuda.is_available():
    args.device = torch.device('cuda', args.cuda_id)
else:
    args.device = torch.device('cpu')

model = model_dict[args.model].EXE(args, load=False)
model.run()