import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from NeuralGraph.dataset import AllData_pk
from NeuralGraph.model import GCN, GCN_FIN, GAT, GAT_FIN, GCH, GCH_FIN
from NeuralGraph.util import Timer, setup_seed
from NeuralGraph.pickle_out import pickle_out
from collections import Counter

algorithm_dict = {
    'GCN': GCN,
    'GCN_FIN': GCN_FIN,
    'GAT': GAT,
    'GAT_FIN': GAT_FIN,
    'GCH': GCH,
    'GCH_FIN': GCH_FIN
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dim_m', type=int, default=128)
parser.add_argument('--dim_p', type=int, default=100)
parser.add_argument('--algorithm', type=str, default='GCN')
parser.add_argument('--input_dir', type=str, default='pickle/pickle_22_CP3A4')
parser.add_argument('--output_dir', type=str, default='output')
args = parser.parse_args()

def run(pickle_from_dir, save_dir, config):
    SEED = config['SEED']
    BATCH_SIZE = config['BATCH_SIZE']
    N_EPOCH = config['N_EPOCH']
    HIDDEN_DIM_FOR_MOLECULAR = config['HIDDEN_DIM_FOR_MOLECULAR']
    HIDDEN_DIM_FOR_PROTEIN = config['HIDDEN_DIM_FOR_PROTEIN']

    algorithm_name = config['algorithm_name']

    setup_seed(SEED)
    print('\nTRAIN START!!!')
    print('\nconfig: ')
    print(config)
    with Timer() as t2:
        train_set, valid_set = pickle_out(start=0, amount=1000, random_state=None, save_dir=pickle_from_dir)
        print('train:',Counter(train_set[5].view(-1).cpu().numpy().tolist()))
        print('valid:',Counter(valid_set[5].view(-1).cpu().numpy().tolist()))
        train_set, valid_set = AllData_pk(train_set), AllData_pk(valid_set)
        print('pickle:')

    with Timer() as t3:
        print(len(train_set), len(valid_set))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        print('data load:')

    with Timer() as t4:
        protein_id = pickle_from_dir.split('/')[-1][7:]
        model_save_path = '{}/{}/{}'.format(save_dir, algorithm_name, protein_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        res_save_path = '{}/{}.res'.format(model_save_path, protein_id)
        algorithm = algorithm_dict[algorithm_name]

        print('\nThe model_save_path is: {}'.format(model_save_path))
        print('The res_save_path is: {}'.format(res_save_path))
        print('The selected algorithm is: {}'.format(algorithm))

        net = algorithm(hid_dim_m=HIDDEN_DIM_FOR_MOLECULAR, hid_dim_p=HIDDEN_DIM_FOR_PROTEIN, n_class=1, save_path=res_save_path)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path=model_save_path)
        print('model:')

if __name__ == '__main__':
    config = {
        'SEED': args.seed,
        'BATCH_SIZE': args.batch_size,
        'N_EPOCH': args.epoch,
        'HIDDEN_DIM_FOR_MOLECULAR': args.dim_m,
        'HIDDEN_DIM_FOR_PROTEIN': args.dim_p,
        'algorithm_name': args.algorithm
    }
    run(args.input_dir, args.output_dir, config)