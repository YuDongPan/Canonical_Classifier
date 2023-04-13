# Designer:Yudong Pan
# Coder:God's hand
# Time:2023/4/13 19:23
import argparse
import numpy as np
import Utils.EEGDataset as EEGDataset
from sklearn.metrics import confusion_matrix
from Model import MSI
from Utils import Ploter

'''                Fs    Nc     Nh     Nf     Ns 
           Dial:  256    8     180     12    10  
'''
parser = argparse.ArgumentParser()

'''
Dial SSVEP Dataset
'''
parser.add_argument('--dataset', type=str, default='Dial', help="12-class dataset")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=512, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--UD', type=int, default=-1, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
parser.add_argument('--ratio', type=int, default=-1, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")

opt = parser.parse_args()


# 2、Start Train
final_acc_list = []
for fold_num in range(opt.Kf):
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, opt.Ns + 1):
        train_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.0, mode="train")
        test_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.0, mode="test")
        # train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")
        # test_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        # squeeze the empty dimension
        eeg_train = eeg_train.squeeze(1).numpy()
        eeg_test = eeg_test.squeeze(1).numpy()

        # -----------------------------------------------------------------------------------------------------------
        print("eeg_train.shape:", eeg_train.shape)
        print("eeg_test.shape:", eeg_test.shape)
        msi = MSI.MSI_Base(opt=opt)
        targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                   10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

        labels, predicted_labels = msi.msi_classify(targets, eeg_test, num_harmonics=3)
        c_mat = confusion_matrix(labels, predicted_labels)
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
        print(f'Subject: {subject}, Classification Accuracy:{accuracy:.3f}')
        final_valid_acc_list.append(accuracy)

    final_acc_list.append(final_valid_acc_list)

# 3、Plot result
Ploter.plot_save_Result(final_acc_list, model_name='MSI', dataset=opt.dataset, UD=opt.UD, ratio=opt.ratio,
                        win_size=str(opt.ws), text=True)