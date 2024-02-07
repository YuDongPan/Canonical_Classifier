# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/1 16:14
import scipy
from scipy import signal
import numpy as np
import pandas as pd
import math
class TRCA():
    def __init__(self, opt, train_dataset, test_dataset):
        self.opt = opt
        self.Fs = opt.Fs
        self.T = int(self.Fs * opt.ws)
        self.Nm = self.opt.Nm
        self.Nc = self.opt.Nc
        self.Nf = self.opt.Nf
        self.dataset = self.opt.dataset
        self.is_ensemble = self.opt.is_ensemble

        self.train_data = train_dataset[0].reshape(self.Nf, -1, self.Nc, self.T)  # (Nh, Nc, T) -> (Nf, Nb, Nc, T)
        self.train_label = train_dataset[1].reshape(self.Nf, -1)  # (Nh, N) -> (Nf, Nb, 1)
        self.test_data = test_dataset[0].reshape(self.Nf, -1, self.Nc, self.T)
        self.test_label = test_dataset[1].reshape(self.Nf, -1)

    def load_data(self):
        self.train_data = self.filter_bank(self.train_data)
        self.test_data = self.filter_bank(self.test_data)
        # print("train_data.shape:", self.train_data.shape)
        # print("test_data.shape:", self.test_data.shape)

    def filter_bank(self, eeg):
        result = np.zeros((self.Nf, self.Nm, self.Nc, self.T, eeg.shape[1]))

        nyq = self.Fs / 2
        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        highcut_pass, highcut_stop = 80, 90

        gpass, gstop, Rp = 3, 40, 0.5
        for i in range(self.Nm):
            Wp = [passband[i] / nyq, highcut_pass / nyq]
            Ws = [stopband[i] / nyq, highcut_stop / nyq]
            [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
            [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
            data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
            result[:, i, :, :, :] = np.transpose(data, (0, 2, 3, 1))

        return result

    def train_trca(self, eeg):
        [num_targs, _, num_chans, num_smpls, _] = eeg.shape
        trains = np.zeros((num_targs, self.Nm, num_chans, num_smpls))
        W = np.zeros((self.Nm, num_targs, num_chans))
        for targ_i in range(num_targs):
            eeg_tmp = eeg[targ_i, :, :, :, :]
            for fb_i in range(self.Nm):
                traindata = eeg_tmp[fb_i, :, :, :]
                trains[targ_i, fb_i, :, :] = np.mean(traindata, 2)
                w_tmp = self.trca(traindata)
                W[fb_i, targ_i, :] = np.real(w_tmp[:, 0])

        return trains, W

    def trca(self, eeg):
        [num_chans, num_smpls, num_trials] = eeg.shape
        S = np.zeros((num_chans, num_chans))
        for trial_i in range(num_trials - 1):
            x1 = eeg[:, :, trial_i]
            x1 = x1 - np.expand_dims(np.mean(x1, 1), 1).repeat(x1.shape[1], 1)
            for trial_j in range(trial_i + 1, num_trials):
                x2 = eeg[:, :, trial_j]
                x2 = x2 - np.expand_dims(np.mean(x2, 1), 1).repeat(x2.shape[1], 1)
                S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

        UX = eeg.reshape(num_chans, num_smpls * num_trials)
        UX = UX - np.expand_dims(np.mean(UX, 1), 1).repeat(UX.shape[1], 1)
        Q = np.matmul(UX, UX.T)
        W, V = scipy.sparse.linalg.eigs(S, 6, Q)
        return V

    def test_trca(self, eeg, trains, W):
        num_trials = eeg.shape[4]
        if self.Nm == 1:
            fb_coefs = [i for i in range(1, self.Nm + 1)]
        else:
            fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        fb_coefs = np.array(fb_coefs)
        results = np.zeros((self.Nf, num_trials))

        for targ_i in range(self.Nf):
            test_tmp = eeg[targ_i, :, :, :, :]
            r = np.zeros((self.Nm, self.Nf, num_trials))

            for fb_i in range(self.Nm):
                testdata = test_tmp[fb_i, :, :, :]

                for class_i in range(self.Nf):
                    traindata = trains[class_i, fb_i, :, :]
                    if not self.is_ensemble:
                        w = W[fb_i, class_i, :]
                    else:
                        w = W[fb_i, :, :].T
                    for trial_i in range(num_trials):
                        testdata_w = np.matmul(testdata[:, :, trial_i].T, w)
                        traindata_w = np.matmul(traindata[:, :].T, w)
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r[fb_i, class_i, trial_i] = r_tmp[0, 1]

            rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)

            tau = np.argmax(rho, axis=0)
            results[targ_i, :] = tau

        return results

    def cal_itr(self, n, p, t):
        if p < 0 or 1 < p:
            print('Accuracy need to be between 0 and 1.')
            exit()
        elif p < 1 / n:
            print('The ITR might be incorrect because the accuracy < chance level.')
            itr = 0
        elif p == 1:
            itr = math.log2(n) * 60 / t
        else:
            itr = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t
        return itr

    def fit(self):
        # Training stage
        # print(traindata.shape)
        trains, W = self.train_trca(self.train_data)

        # Test stage
        # print(testdata.shape)
        estimated = self.test_trca(self.test_data, trains, W)

        # Evaluation
        is_correct = (estimated == self.test_label)
        is_correct = np.array(is_correct).astype(int)

        test_acc = np.mean(is_correct)

        return test_acc



