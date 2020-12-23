import torch as T
from torch import nn
from torch.nn import functional as F
from .layer import GraphConv, GraphConv_m, GraphConv_p, GraphPool, GraphOutput, GraphOutput_p
from .layer import GraphAttention, GraphCore, Interaction_Aggregation
import numpy as np
from torch import optim
import time
from .util import dev
from sklearn.metrics import accuracy_score,roc_auc_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = T.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return T.mean(F_loss)
        else:
            return F_loss

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)

def save_empty(save_path):
    col_names = ['epoch_num', 'epoch_time', 'loss_train', 'loss_valid', 'acc_train', 'acc_valid', 'auc_train', 'auc_valid']
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(col_names) + '\n')

def save_records(records_lst, save_path):
    records_lst = ['\t'.join([str(item) for item in line]) + '\n' for line in records_lst]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(records_lst)

def save_records_a(epoch_record, save_path):
    epoch_record = [str(item) for item in epoch_record]
    epoch_record = '\t'.join(epoch_record) + '\n'
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(epoch_record)

class GCN(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GCN, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphConv_p(input_dim=480, conv_width=200)
        self.gcn_pro_2 = GraphConv_p(input_dim=200, conv_width=100)
        self.gop = GraphOutput(input_dim=134, output_dim=hid_dim_m)
        self.gop_p = GraphOutput_p(input_dim=100, output_dim=hid_dim_p)
        self.FC_layer_1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.FC_layer_2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))
        fp_m = self.gop(m_atoms, m_bonds, m_edges)

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))
        fp_p = self.gop_p(p_atoms, p_edges)

        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.FC_layer_1(fp)
        fp = self.FC_layer_2(fp)
        out = T.sigmoid(self.output_layer(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class GCN_FIN(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GCN_FIN, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphConv_p(input_dim=480, conv_width=200)
        self.gcn_pro_2 = GraphConv_p(input_dim=200, conv_width=100)
        self.Interaction_Aggregation = Interaction_Aggregation(128, 100, 0.1)
        self.FC_layer_1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.FC_layer_2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))

        fp_m, fp_p = self.Interaction_Aggregation(m_atoms, p_atoms)
        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.FC_layer_1(fp)
        fp = self.FC_layer_2(fp)
        out = T.sigmoid(self.output_layer(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class GAT(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GAT, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphAttention(in_features=480, out_features=200, alpha=0.2)
        self.gcn_pro_2 = GraphAttention(in_features=200, out_features=100, alpha=0.2)
        self.gop = GraphOutput(input_dim=134, output_dim=hid_dim_m)
        self.gop_p = GraphOutput_p(input_dim=100, output_dim=hid_dim_p)
        self.fc1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))
        fp_m = self.gop(m_atoms, m_bonds, m_edges)

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))
        fp_p = self.gop_p(p_atoms, p_edges)

        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.fc1(fp)
        fp = self.fc2(fp)
        out = T.sigmoid(self.fc3(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class GAT_FIN(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GAT_FIN, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphAttention(in_features=480, out_features=200, alpha=0.2)
        self.gcn_pro_2 = GraphAttention(in_features=200, out_features=100, alpha=0.2)
        self.Interaction_Aggregation = Interaction_Aggregation(128, 100, 0.1)
        self.fc1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))

        fp_m, fp_p = self.Interaction_Aggregation(m_atoms, p_atoms)
        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.fc1(fp)
        fp = self.fc2(fp)
        out = T.sigmoid(self.fc3(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class GCH(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GCH, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphCore(in_features=480, out_features=200, alpha=0.2, ratio=0.5, khop=2)
        self.gcn_pro_2 = GraphCore(in_features=200, out_features=100, alpha=0.2, ratio=0.5, khop=2)
        self.gop = GraphOutput(input_dim=134, output_dim=hid_dim_m)
        self.gop_p = GraphOutput_p(input_dim=100, output_dim=hid_dim_p)
        self.fc1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))
        fp_m = self.gop(m_atoms, m_bonds, m_edges)

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))
        fp_p = self.gop_p(p_atoms, p_edges)

        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.fc1(fp)
        fp = self.fc2(fp)
        out = T.sigmoid(self.fc3(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class GCH_FIN(nn.Module):
    def __init__(self, hid_dim_m, hid_dim_p, n_class, save_path):
        super(GCH_FIN, self).__init__()
        self.gcn_mol_1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn_mol_2 = GraphConv(input_dim=134, conv_width=128)
        self.gcn_pro_1 = GraphCore(in_features=480, out_features=200, alpha=0.2, ratio=0.5, khop=2)
        self.gcn_pro_2 = GraphCore(in_features=200, out_features=100, alpha=0.2, ratio=0.5, khop=2)
        self.Interaction_Aggregation = Interaction_Aggregation(128, 100, 0.1)
        self.fc1 = nn.Linear(hid_dim_m + hid_dim_p, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)
        self.to(dev)
        self.save_path = save_path

    def forward(self, m_atoms, m_bonds, m_edges, p_atoms, p_edges):
        m_atoms = F.relu(self.gcn_mol_1(m_atoms, m_bonds, m_edges))
        m_atoms = F.relu(self.gcn_mol_2(m_atoms, m_bonds, m_edges))

        p_atoms = F.relu(self.gcn_pro_1(p_atoms, p_edges))
        p_atoms = F.relu(self.gcn_pro_2(p_atoms, p_edges))

        fp_m, fp_p = self.Interaction_Aggregation(m_atoms, p_atoms)
        fp = T.cat([fp_m, fp_p], dim=1)
        fp = self.fc1(fp)
        fp = self.fc2(fp)
        out = T.sigmoid(self.fc3(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=5, lr=1e-5):
        criterion = FocalLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0

        save_empty(self.save_path)
        for epoch in range(epochs):
            t0 = time.time()
            loss_train, acc_train, auc_train = 0.0, 0.0, 0.0
            for Ab, Bb, Eb, Nb, Vb, yb in loader_train:
                Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Nb/300.0, Vb/30.0)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss_train += loss
                acc_train += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
                auc_train += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
                loss.backward()
                optimizer.step()
            loss_train, acc_train, auc_train = loss_train/len(loader_train), acc_train/len(loader_train), auc_train/len(loader_train)
            loss_valid, acc_valid, auc_valid = self.evaluate(loader_valid)
            epoch_time = time.time() - t0

            epoch_record = [epoch, float(epoch_time), float(loss_train.cpu().detach().numpy().tolist()), float(loss_valid), float(acc_train), float(acc_valid), float(auc_train), float(auc_valid)]

            save_records_a(epoch_record, self.save_path)

            print('[Epoch:%d/%d] %.1fs loss_train: %.3f loss_valid: %.3f auc_train: %.3f auc_valid: %.3f' % (epoch, epochs, epoch_time, loss_train, loss_valid, auc_train , auc_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop and auc_valid > 0.85: break

        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss, acc, auc = 0.0, 0.0, 0.0
        criterion = FocalLoss()
        for Ab, Bb, Eb, Nb, Vb, yb in loader:
            Ab, Bb, Eb, Nb, Vb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), Nb.to(dev), Vb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Nb, Vb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
            acc += accuracy_score((y_>0.5).type_as(yb).cpu(),yb.cpu())
            auc += roc_auc_score_FIXED(yb.cpu().detach(),y_.cpu().detach())
        return loss / len(loader), acc / len(loader), auc / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb, Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score