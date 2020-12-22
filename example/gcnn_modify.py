# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from NeuralGraph.dataset import AllData_pk
from NeuralGraph.model import GraphConvAutoEncoder, QSAR
from NeuralGraph.util import Timer
from NeuralGraph.pickle_out import pickle_out
from collections import Counter
import numpy as np


def main():
    print('\nTRAIN START!!!')
    with Timer() as t2:
        train_set, valid_set = pickle_out(start=0, amount=1, random_state=None)
        # print(train_set[5].shape)
        # print(valid_set[5].shape)
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
        # net = GraphConvAutoEncoder(hid_dim_m=128, hid_dim_p=128, n_class=2)
        # net = net.fit(train_loader, epochs=100)
        net = QSAR(hid_dim_m=216, hid_dim_p=512, n_class=2)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path='output/gcnn')
        print('model:')


if __name__ == '__main__':
    BATCH_SIZE = 128 # 128
    N_EPOCH = 200
    main()
