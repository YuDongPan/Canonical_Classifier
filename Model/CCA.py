# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/1/28 0:13
import numpy as np
from sklearn.cross_decomposition import CCA
class CCA_Base():
    def __init__(self, opt):
        super(CCA_Base, self).__init__()
        self.Nh = opt.Nh
        self.Fs = opt.Fs
        self.Nf = opt.Nf
        self.ws = opt.ws
        self.Nc = opt.Nc
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

    def cca_classify(self, targets, test_data, num_harmonics=3, train_data=None, template=False):
        if template:
            reference_signals = self.get_Template_Signal(train_data, targets)
        else:
            reference_signals = self.get_Reference_Signal(num_harmonics, targets)

        print("segmented_data.shape:", test_data.shape)
        print("reference_signals.shape:", reference_signals.shape)

        predicted_class = []
        labels = []
        num_segments = test_data.shape[0]
        num_perCls = num_segments // reference_signals.shape[0]

        for segment in range(0, num_segments):
            labels.append(segment // num_perCls)
            result = self.find_correlation(1, test_data[segment], reference_signals)
            predicted_class.append(np.argmax(result) + 1)

        labels = np.array(labels) + 1
        predicted_class = np.array(predicted_class)
        return labels, predicted_class
