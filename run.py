# code adapted from https://github.com/yaohungt/Multimodal-Transformer  
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train,train_urfunny
from humor_dataloader import HumorDataset,load_pickle


parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='Superformer',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--device', type=str, default='cpu', help='device')

# Tasks
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.2,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.3,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout')


# Architecture
parser.add_argument('--layers', type=int, default=4,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--d_model', type=int, default=32,
                    help='d_model')
parser.add_argument('--S', type=float, default=5, help='S')
parser.add_argument('--r', type=list, default=[8,4,3], help='r')
parser.add_argument('--shift_mode', type=dict, 
                    default=dict(I=['S,P,R'],X=['S'],S=['S'],C=[1,0.25,0.05]),
                    help='shift mode')
parser.add_argument('--use_fast', type=bool, default=False, help='use fast attention')
parser.add_argument('--use_dense', type=bool, default=False, help='use dense attention')


# Tuning
parser.add_argument('--batch_size', type=int, default=250, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='test2',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()


if not os.path.exists('./pre_trained_models'): os.makedirs('./pre_trained_models')
args.data_path='../datasets/Archive'
args.dataset='mosei_senti'
args.aligned=False


torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())

use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################
#                  #
# Load the dataset #
#                  #
####################

print("Start loading the data....")

if args.dataset=='urfunny':
    path='../datasets/UR-FUNNY/'
    data_folds_file= path+"data_folds.pkl"
    data_folds= load_pickle(data_folds_file)

    max_context_len=8
    max_sen_len=40
    train_set = HumorDataset(data_folds['train'],path,max_context_len,max_sen_len)
    dev_set = HumorDataset(data_folds['dev'],path,max_context_len,max_sen_len)
    test_set = HumorDataset(data_folds['test'],path,max_context_len,max_sen_len)
else:
    train_set = get_data(args, dataset, 'train')
    dev_set = get_data(args, dataset, 'valid')
    test_set = get_data(args, dataset, 'test')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

print('Finish loading the data....')

###################
#                 #
# Hyperparameters #
#                 #
###################

hyp_params = args
if args.dataset=='urfunny': hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 300, 81, 75
else: 
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_set.get_dim()
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_set.get_seq_len()
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_set), len(dev_set), len(test_set)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = 1
hyp_params.criterion = 'L1Loss'

if args.dataset=='urfunny': trainer=train_urfunny.initiate
else: trainer=train.initiate

if __name__ == '__main__':
    metric = trainer(hyp_params, train_loader, dev_loader, test_loader, test_only=False)
