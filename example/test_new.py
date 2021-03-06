import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from NeuralGraph.dataset import AllData_pk
from NeuralGraph.model import GCH_FIN, GCH_FIN_cp
from NeuralGraph.util import Timer
from NeuralGraph.pickle_out import pickle_out
from collections import Counter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 设置随机数种子
    setup_seed(0)

    print('\nTRAIN START!!!')
    with Timer() as t2:
        train_set, valid_set = pickle_out(start=0, amount=10, random_state=0)
        train_amount = 5000
        pos_amount, neg_amount = 50, 50 # 5 pos and 5 neg
        pos_lst, neg_lst = [], []

        print('train:',Counter(train_set[5][:train_amount].view(-1).cpu().numpy().tolist()))
        print('valid:',Counter(valid_set[5].view(-1).cpu().numpy().tolist()))
        train_set, valid_set = AllData_pk(train_set), AllData_pk(valid_set)

        print(train_set[0][1].shape)

        for valid_item in valid_set:
            if pos_amount!=0 and valid_item[5].item()==1:
                pos_amount-=1
                pos_lst.append(valid_item)
                continue
            if neg_amount!=0 and valid_item[5].item()==0:
                neg_amount-=1
                neg_lst.append(valid_item)
                continue
            if pos_amount==0 and neg_amount==0:
                break
        print('valid: Counter 0.0:  {}, 1.0: {}'.format(len(pos_lst),len(neg_lst)))
        valid_set = pos_lst + neg_lst
        print('pickle:')
    with Timer() as t3:
        # print('train data: {}, vaild data: {}\n'.format(len(train_set), len(valid_set)))
        print('train data: {}, vaild data: {}\n'.format(train_amount, len(valid_set)))
        t_set = [train_set[i] for i in range(train_amount)]
        train_loader = DataLoader(t_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        print('data load:')
    with Timer() as t4:
        net = GCH_FIN_cp(hid_dim_m=216, hid_dim_p=512, n_class=1)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path='output/gcnn')
        print('model:')


if __name__ == '__main__':
    BATCH_SIZE = 20 # 128
    N_EPOCH = 100
    main()
