# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io
from scipy import signal

class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP12Intra, self).__init__()
        self.train_ratio = train_ratio
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.subject = subject
        self.eeg_data, self.label_data = self.load_Data()
        self.num_trial = self.Nh // self.Nf
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue
                if KFold is not None:
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def load_Data(self):
        subjectfile = scipy.io.loadmat(f'../data/Dial/S{self.subject}.mat')
        samples = subjectfile['eeg']  # (12, 8, 1024, 15)
        eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
        for i in range(1, 12):
            eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
        eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1114, 180) -> (180, 8, 1024)
        eeg_data = np.expand_dims(eeg_data, axis=1)  # (180, 8, 1024) -> (180, 1, 8, 1114)
        eeg_data = torch.from_numpy(eeg_data)
        label_data = np.zeros((180, 1))
        for i in range(12):
            label_data[i * 15:(i + 1) * 15] = i
        label_data = torch.from_numpy(label_data)
        print(eeg_data.shape)
        print(label_data.shape)
        return eeg_data, label_data



