# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/1/28 0:13
import numpy as np
import math
from scipy import signal
from sklearn.cross_decomposition import CCA
class FBCCA():
    def __init__(self, opt):
        super(FBCCA, self).__init__()
        self.Nh = opt.Nh
        self.Fs = opt.Fs
        self.Nf = opt.Nf
        self.ws = opt.ws
        self.Nc = opt.Nc
        self.Nm = opt.Nm
        self.dataset = opt.dataset
        self.T = int(self.Fs * self.ws)

    def get_Reference_Signal(self, num_harmonics, targets):
        reference_signals = []
        t = np.arange(0, (self.T / self.Fs), step=1.0 / self.Fs)
        for f in targets:
            reference_f = []
            for h in range(1, num_harmonics + 1):
                reference_f.append(np.sin(2 * np.pi * h * f * t)[0:self.T])
                reference_f.append(np.cos(2 * np.pi * h * f * t)[0:self.T])
            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals

    def get_Template_Signal(self, X, targets):
        reference_signals = []
        num_per_cls = X.shape[0] // self.Nf
        for cls_num in range(len(targets)):
            reference_f = X[cls_num * num_per_cls:(cls_num + 1) * num_per_cls]
            reference_f = np.mean(reference_f, axis=0)
            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals

    def find_correlation(self, n_components, X, Y):
        cca = CCA(n_components)
        corr = np.zeros(n_components)
        num_freq = Y.shape[0]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            matched_X = X

            cca.fit(matched_X.T, Y[freq_idx].T)
            # cca.fit(X.T, Y[freq_idx].T)
            x_a, y_b = cca.transform(matched_X.T, Y[freq_idx].T)
            for i in range(0, n_components):
                corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
                result[freq_idx] = np.max(corr)

        return result

    def filter_bank(self, eeg):
        result = np.zeros((eeg.shape[0], self.Nm, eeg.shape[-2], self.T))

        nyq = self.Fs / 2
        if self.dataset == 'Direction':
            passband = [4, 10, 16, 22, 28, 34, 40]
            stopband = [2, 6, 10, 16, 22, 28, 34]
            highcut_pass, highcut_stop = 40, 50

        else:
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
            result[:, i, :, :] = data

        return result

    def fbcca_classify(self, targets, test_data, num_harmonics=3, train_data=None, template=False):
        if template:
            train_data = self.filter_bank(train_data)
            reference_signals = np.zeros((self.Nf, self.Nm, self.Nc, self.T))
            for fb_i in range(0, self.Nm):
                reference_signals[:, fb_i] = self.get_Template_Signal(train_data[:, fb_i], targets)
        else:
            reference_signals = self.get_Reference_Signal(num_harmonics, targets)

        test_data = self.filter_bank(test_data)

        print("segmented_data.shape:", test_data.shape)
        print("reference_signals.shape:", reference_signals.shape)

        predicted_class = []
        labels = []
        num_segments = test_data.shape[0]
        num_perCls = num_segments // reference_signals.shape[0]

        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]  # w(n) = n^(-0.5) + 1.25
        for segment in range(0, num_segments):
            labels.append(segment // num_perCls)
            result = np.zeros(self.Nf)
            # result = ¦² w(n) * (¦Ñ(k))^2
            for fb_i in range(0, self.Nm):
                x = test_data[segment, fb_i]
                y = reference_signals[:, fb_i] if template else reference_signals
                w = fb_coefs[fb_i]

                result += (w * (self.find_correlation(1, x, y) ** 2))

            predicted_class.append(np.argmax(result) + 1)

        labels = np.array(labels) + 1
        predicted_class = np.array(predicted_class)
        return labels, predicted_class
