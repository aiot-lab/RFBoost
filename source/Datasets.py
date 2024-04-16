from ast import arg
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score
from sklearn.utils import resample
import scipy.io as sio
from utils import *
from augment import *
import torch.nn.functional as F
import tqdm
import mat73

from collections import Counter

class RFBoostDataset(Dataset):
    def __init__(self, args, path_file, part, config=None, slice_idx=None):
        self.part = part
        self.args = args
        self.time_aug = args.time_aug
        self.freq_aug_list = []
        self.freq_args_list = []
        
        if args.freq_aug != [] and args.freq_aug != ['']:
            # e.g. ["kmeans,4", "ms-top,4"]
            for freq_aug_one in args.freq_aug:
                # "kmeans,4"
                fda_policy, freq_args = freq_aug_one.split(',')
                freq_args = int(freq_args)
                for i in range(freq_args):
                    self.freq_aug_list.append(fda_policy)
                    self.freq_args_list.append((i, freq_args))
        else:
            self.freq_args_list = []

        # Augmentation parameters
        self.space_aug = args.space_aug

        self.n_fda = len(self.freq_aug_list)
        self.n_tda = len(self.time_aug)
        self.n_sda = len(self.space_aug)

        self.augment = Augmentation(args.default_stft_window, window_step=10)
        self.aug_ratio = self.n_fda + self.n_tda + self.n_sda
        
        if self.part in ["test"] and self.args.exp_test.startswith("rx"):
            # rx_sel: rx-0,2,4
            self.rx_candidate = self.args.exp_test.split('-')[1].split(',')
        elif self.part in ["train", "valid"] and self.args.exp_train_val.startswith("rx"):
            # rx_sel: rx-0,2,4
            self.rx_candidate = self.args.exp_train_val.split('-')[1].split(',')
        else:
            self.rx_candidate = ["all"]
                        
        if self.args.dataset == "widar3":
            # args.data_path = ""
            root_folder = "./widar3/{}/".format(args.data_path)
            # Hard
            # which = "fold0"
            self.records = np.load(root_folder+"{}_filename.npy".format(self.part))
            self.labels = np.load(root_folder+"{}_label.npy".format( self.part), allow_pickle=True)

    def apply_slice(self, slice_idx):
        if slice_idx is None:
            return 

        self.records = [self.records[i] for i in slice_idx]
        # self.data_paths = [self.data_paths[i] for i in slice_idx]
        self.labels = [self.labels[i] for i in slice_idx]
        self.ms = [self.ms[i] for i in slice_idx]
        
    def label_dist_str(self):
        label_conter = Counter(self.labels)
        dist_str = "["
        # sort by label
        for label, cnt in sorted(label_conter.items(), key=lambda x: int(x[0])):
            dist_str += "{}:{:} ".format(label, cnt)
        dist_str += "]"
        return dist_str
    def index_map(self, index, shape):
        # shape: [N, A, Rx]
        # return N_i, A_i, Rx_i of index
        N, A, Rx = shape
        N_i = index // (A*Rx)
        A_i = (index % (A*Rx)) // Rx
        Rx_i = (index % (A*Rx)) % Rx
        
        return N_i, A_i, Rx_i

    def __getitem__(self, index):
        """
        index mapping to the original data
        len(time_aug) + len(freq_aug) + len(space_aug) = aug_ratio - 1
        0...         a1-1 ..      a1+a2-1...        a1+a2+a3-1... a1+a2+a3(original)

        0   1   2   3   ...   aug_ratio-1
        aug             ...
        2*aug

        ...
        (n-1)*aug     ...     n*aug_ratio-1

        """
        # if augmentation is enabled
        if self.args.exp.startswith("imb-"):
            # don't use augmentation for all
            file_idx = index
            # original data
            aug_idx = self.aug_ratio
        else:
            file_idx, aug_idx, rx_idx = self.index_map(index, [len(self.records), self.aug_ratio, len(self.rx_candidate)])

        # Get Spectrogram
        try:
            if self.args.dataset.startswith("widar3"):
                path = self.records[file_idx]
                label = self.labels[file_idx]
                try:
                    if self.args.version == "norm-filter":
                        data_path = "../dataset/NPZ-pp/{}/{}.npz".format(self.args.version, path)
                        ms_path = "../dataset/NPZ-pp/{}-ms/{}.mat".format(self.args.version, path)
                        csi_data, ms = np.load(data_path)['data'], sio.loadmat(ms_path)['ms']
                    elif self.args.version == "norm-filter-2024":
                        data_path = "../dataset/NPZ-pp/{}/{}.mat".format(self.args.version, path)
                        data_packed = sio.loadmat(data_path)
                        csi_data, ms = data_packed['data'], data_packed['ms']
                    # label start from 0
                    label = int(label) - 1

                except Exception as e:
                    self.args.log(e)
                    self.args.log("Error: {}".format(data_path))
                    raise Exception("Error: {}".format(data_path))

                if self.rx_candidate is not None and self.rx_candidate[0] != "all":
                    # rx_sel: rx-0,2,4
                    rx_sel = int(self.rx_candidate[rx_idx])
                    
                    csi_data = csi_data[:, :, rx_sel][:, :, np.newaxis]
                    ms = ms[:, :, rx_sel][:, :, np.newaxis]

                # [T, F, Rx] -> [Rx, T, F]
                csi_data = np.transpose(csi_data, (2, 0, 1))
                if ms is not None:
                # if ms has only 1 dim
                    if ms.ndim == 1:
                        ms = np.expand_dims(ms, axis=0)
                    else:
                        # # [W, F, Rx] -> [Rx, F, W]
                        ms = np.transpose(ms, (2, 1, 0))

            # with record_function("Get_DFS"):
            # Augmentation
            if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
                if aug_idx < self.n_tda:
                    tda_idx = aug_idx
                    dfs = [self.augment.time_augment(csi_data[i], ms[i], self.time_aug[tda_idx], args=self.args, agg_type="pca")[2] for i in range(csi_data.shape[0])] 
                elif aug_idx < self.n_tda + self.n_fda:
                    fda_idx = aug_idx - self.n_tda 
                    dfs = [self.augment.frequency_augment(csi_data[i], ms[i], self.freq_aug_list[fda_idx], self.freq_args_list[fda_idx][1], fda_th=self.freq_args_list[fda_idx][0], file_th=file_idx, rx_th=i, args=self.args)[2] for i in range(csi_data.shape[0])]
                elif aug_idx < self.n_tda + self.n_fda + self.n_sda:
                    sda_idx = aug_idx - self.n_tda - self.n_fda
                    dfs = [self.augment.space_augment(csi_data[i], ms[i], self.space_aug[sda_idx], args=self.args)[2] for i in range(csi_data.shape[0])]
                else:
                    # error
                    raise Exception("Error: {}".format(data_path))
            else:
                fda_idx = aug_idx - self.n_tda 
                dfs = [self.augment.frequency_augment(csi_data[i], ms[i], "all", self.freq_args_list[fda_idx][1], fda_th=self.freq_args_list[fda_idx][0], file_th=file_idx, rx_th=i, args=self.args)[2] for i in range(csi_data.shape[0])]

            if np.array(dfs).shape.__len__() != 3:
                # (1, 90, 121, 139) -> (1x90, 121, 139)
                dfs = np.array(dfs)
                dfs = dfs.reshape((dfs.shape[0]*dfs.shape[1], dfs.shape[2], dfs.shape[3]))
            
            # [Rx, F, W]
            dfs = np.array([pad_and_downsample(d, self.args.input_size) for d in dfs])

            if self.args.model == 'Widar3':
                # [Rx, F, W] -> [W, Rx, F] 
                dfs = dfs.transpose((2, 0, 1))
                # [W, Rx, F] -> [W, 1, Rx, F] extend_dim
                dfs = np.expand_dims(dfs, axis=1)
            elif self.args.model in ["ResNet18", "AlexNet"]:
                # [Rx, F, W] -> [Rx, W, F]
                dfs = dfs.transpose((0, 2, 1))
            elif self.args.model == "RFNet":
                # [Rx, F, W] -> [W, Rx, F] 
                dfs = dfs.transpose((2, 0, 1))
                # [W, Rx, F] -> [W, Rx*F] 
                dfs = dfs.reshape((dfs.shape[0], -1))

            elif self.args.model == "CNN_GRU":
                # [W, Rx*F] -> [W, 1, Rx*F]
                dfs = np.expand_dims(dfs, axis=1)

            # normalize
            dfs = (dfs - np.mean(dfs)) / np.std(dfs)

            if np.isnan(dfs).any():
                print('nan')

            return dfs, label
        except Exception as e:
            self.args.log(e)
            self.args.log("Error: {}".format(data_path))
            raise Exception("Error: {}".format(data_path))

    def __len__(self):
        if self.args.exp.startswith("imb-"):
            return len(self.records)

        if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
            return len(self.records) * self.aug_ratio * len(self.rx_candidate)
        else:
            return len(self.records) * self.aug_ratio * len(self.rx_candidate)

def eval(model, eval_loader, epoch, kind, args):
    y_pred = []
    y_true = []
    prob_all =[]
    eval_loss = 0
    iter_conter = 0
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(tqdm.tqdm(eval_loader)):
            iter_conter += 1
            x = x.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True).long()

            out = model(x)
            prob = F.softmax(out, dim=1)
            prob_all.append(prob.cpu().numpy())
            # eval_loss += args.loss_func(out, y)
            loss = args.loss_func(out, y)
            eval_loss += loss.cpu().item()
            
            pred = torch.argmax(out, dim = -1)
            y_pred += pred.cpu().tolist()
            y_true += y.cpu().tolist()

        eval_loss /= iter_conter
        
        C = eval_loader.dataset.aug_ratio
        B = len(y_true) // C    
        # majority voting
        # [B*C] -> [B, C]
        y_pred = np.array(y_pred)[:B*C].reshape((B, C))
        y_true = np.array(y_true)[:B*C].reshape((B, C))
        prob_all = np.concatenate(prob_all, axis=0)
        prob_all = np.array(prob_all)[:B*C,:].reshape((B, C, -1))
        # [B, C] -> [B]
        y_pred = [np.bincount(p).argmax() for p in y_pred]
        y_true = [np.bincount(p).argmax() for p in y_true]
        prob_all = np.array([np.mean(p, axis=0) for p in prob_all])

    if args.num_labels == 2:
        # calculate ROC curve using out
        prob_all = np.concatenate(prob_all, axis=0)
        fpr, tpr, thresholds = roc_curve(y_true, prob_all[:, 1])
        args.log("ROC fpr: {}, tpr: {}, thresholds: {}".format(fpr, tpr, thresholds))
        auc = roc_auc_score(y_true, prob_all[:, 1], average='weighted')
        # fpr80
        fpr80 = fpr[np.where(tpr >= 0.8)[0][0]]
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
        fnr = 1 - tpr
        eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]
        eer_pred = prob_all[:, 1] >= eer_threshold
        y_pred = eer_pred.astype(int)
        auprc = average_precision_score(y_true, prob_all[:, 1], average='weighted')

    eval_acc = accuracy_score(y_true, y_pred)
    if args.num_labels == 2:
        eval_f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        eval_f1 = f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')
    """
    draw a confusion matrix with TF, TN, FP, FN:
    GT\Pred |Positive | Negative
    ------------------+-----------
    True    | TP      | FN          # <- Fall
    False   | FP      | TN          # <- Normal
    
    false alarm rate = FP / (FP + TN)
    miss alarm rate = FN / (FN + TP)
    detection rate = TP / (TP + FN)

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)

    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    false_alarm_rate = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
    miss_alarm_rate = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1)
    detection_rate = 1 - miss_alarm_rate
    if kind == "valid":
        args.log("[epoch={}]Validation Accuracy : {:.7} Macro F1 : {:.7} Loss : {:.7}\n".
            format(epoch, str(eval_acc), str(eval_f1), str(eval_loss)))
        args.writer.add_scalar('accuracy/valid', eval_acc, epoch)
        args.writer.add_scalar('f1/valid', eval_f1, epoch)
        args.writer.add_scalar('loss/valid', eval_loss, epoch)
        args.writer.add_scalar('FAR/valid', false_alarm_rate, epoch)
        args.writer.add_scalar('MAR/valid', miss_alarm_rate, epoch)
        if args.num_labels == 2:
            args.writer.add_scalar('AUC/valid', auc, epoch)
            args.writer.add_scalar('AUPRC/valid', auprc, epoch)
            # detection_rate
            args.writer.add_scalar('DET/valid', detection_rate, epoch)
            # args.writer.add_scalar('EER/valid', eer, epoch)
            args.writer.add_scalar('FPR80/valid', fpr80, epoch)
            args.writer.add_scalar('FPR95/valid', fpr95, epoch)

    elif kind == "test":
        args.log("[epoch={}]Test Accuracy : {:.7} Macro F1 : {:.7} Loss : {:.7}\n".
            format(epoch, str(eval_acc), str(eval_f1), str(eval_loss)))
        args.writer.add_scalar('accuracy/test', eval_acc, epoch)
        args.writer.add_scalar('f1/test', eval_f1, epoch)
        args.writer.add_scalar('loss/test', eval_loss, epoch)
        args.writer.add_scalar('FAR/test', false_alarm_rate, epoch)
        args.writer.add_scalar('MAR/test', miss_alarm_rate, epoch)
        if args.num_labels == 2:
            args.writer.add_scalar('AUC/test', auc, epoch)
            args.writer.add_scalar('AUPRC/test', auprc, epoch)
            # args.writer.add_scalar('EER/valid', eer, epoch)
            args.writer.add_scalar('DET/test', detection_rate, epoch)
            args.writer.add_scalar('FPR80/test', fpr80, epoch)
            args.writer.add_scalar('FPR95/test', fpr95, epoch)


    # confusion matrix
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_labels)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    return eval_loss, eval_acc, eval_f1

def train(model, train_loader, optimizer, epoch, args):
    args.w_all = []
    y_pred = []
    y_true = []
    epoch_loss = 0
    iter_conter = 0
    x_counter = 0
    for i, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        iter_conter += 1
        model.train()

        x = x.cuda(non_blocking=True).float()
        y = y.cuda(non_blocking=True).long()
        
        # [B, W, Rx*F]
        # x = x.permute(0, 3, 1, 2).contiguous().view(x.shape[0], -1, x.shape[1] * x.shape[2])

        out = model(x)            
        loss = args.loss_func(out, y)
        if np.isnan(loss.item()):
            print("Loss is NaN")
        #     exit()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        epoch_loss += loss.cpu().item()

        pred = torch.argmax(out, dim = -1)
        y_pred += pred.cpu().tolist()
        y_true += y.cpu().tolist()

        x_counter += len(x)
        if (i != 0 and (i+1) % (len(train_loader.dataset)//4*args.batch_size) == 0) or x_counter == (len(train_loader.dataset)-1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch, x_counter, len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

    epoch_loss /= iter_conter
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')

    args.writer.add_scalar('loss/train', epoch_loss, epoch)
    args.writer.add_scalar('accuracy/train', train_acc, epoch)
    args.writer.add_scalar('f1/train', train_f1, epoch)

    args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    args.log("End of Epoch : " + str(epoch) + " Loss(avg) : " + str(epoch_loss))
