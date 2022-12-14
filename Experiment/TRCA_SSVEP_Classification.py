# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/1 22:04
import argparse
from Utils import EEGDataset
from Model import TRCA
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
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=1024, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--is_ensemble', type=int, default=0, help="TRCA or eTRCA")
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
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        # squeeze the empty dimension
        eeg_train = eeg_train.squeeze(1).numpy()
        label_train = label_train.numpy()

        eeg_test = eeg_test.squeeze(1).numpy()
        label_test = label_test.numpy()

        trca = TRCA.TRCA(opt, (eeg_train, label_train), (eeg_test, label_test))
        trca.load_data()
        test_acc = trca.fit()
        print(f'Subject: {subject}, test_acc:{test_acc:.3f}')
        final_valid_acc_list.append(test_acc)
        # exit()

    final_acc_list.append(final_valid_acc_list)


# 3、Plot result
Ploter.plot_save_Result(final_acc_list, model_name='TRCA', dataset=opt.dataset, UD=opt.UD, ratio=opt.ratio,
                        win_size=str(opt.ws), text=True)