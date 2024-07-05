# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/4/2 16:56
import argparse
import numpy as np
from Utils import EEGDataset
from Model import TDCA
from Utils import Ploter

# 1、Define parameters of eeg
'''               Fs    Nc    Nh    Nf    Ns   Nm   low   high
      Direction:  100    10   100    4    54    4    4     40
           Dial:  256    8    180    12   10    4    6     80
'''
parser = argparse.ArgumentParser()

'''Dial Dataset'''
parser.add_argument('--dataset', type=str, default='Dial', help="12-class dataset")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nm', type=int, default=1, help="number of bank")
parser.add_argument('--lagging_len', type=int, default=5, help="lagging length")
parser.add_argument('--n_components', type=int, default=1, help="number of components")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=1024, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--UD', type=int, default=0, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
parser.add_argument('--ratio', type=int, default=3, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")

opt = parser.parse_args()
# 2、Start training
final_acc_list = []
for fold_num in range(opt.Kf):
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, opt.Ns + 1):
        train_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.2, mode="train")
        test_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.2, mode="test")
        # train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")
        # test_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * (opt.ws + 0.1))]   # extra window for lagging points
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        # squeeze the empty dimension
        eeg_train = eeg_train.squeeze(1).numpy()
        label_train = label_train.squeeze(-1).numpy()

        eeg_test = eeg_test.squeeze(1).numpy()
        label_test = label_test.squeeze(-1).numpy()

        # pad zero points to eeg test data
        pad_eeg_test = np.zeros((*eeg_test.shape[:2], int((opt.ws + 0.1) * opt.Fs)))
        pad_eeg_test[:, :, :int(opt.ws * opt.Fs)] = eeg_test

        targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                   10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
        tdca = TDCA.TDCA(opt, targets)
        tdca.fit(eeg_train, label_train)
        test_acc = tdca.predict(pad_eeg_test, label_test)

        print(f'Subject: {subject}, test_acc:{test_acc:.3f}')
        final_valid_acc_list.append(test_acc)
        # exit()

    final_acc_list.append(final_valid_acc_list)

# 3、Plot result
Ploter.plot_save_Result(final_acc_list, model_name='TDCA', dataset=opt.dataset, UD=opt.UD, ratio=opt.ratio,
                        win_size=str(opt.ws), text=True)

