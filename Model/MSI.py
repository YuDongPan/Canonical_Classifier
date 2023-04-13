# Designer:Yudong Pan
# Coder:God's hand
# Time:2023/4/13 19:24

import numpy as np

class MSI_Base():
    def __init__(self, opt):
        super(MSI_Base, self).__init__()
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

    def find_Synchronization_Index(self, X, Y):
        num_freq = Y.shape[0]
        num_harm = Y.shape[1]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            y = Y[freq_idx]
            X = X[:] - np.mean(X).repeat(self.T * self.Nc).reshape(self.Nc, self.T)
            X = X[:] / np.std(X).repeat(self.T * self.Nc).reshape(self.Nc, self.T)

            y = y[:] - np.mean(y).repeat(self.T * num_harm).reshape(num_harm, self.T)
            y = y[:] / np.std(y).repeat(self.T * num_harm).reshape(num_harm, self.T)

            c11 = (1 / self.T) * (np.dot(X, X.T))
            c22 = (1 / self.T) * (np.dot(y, y.T))
            c12 = (1 / self.T) * (np.dot(X, y.T))
            c21 = c12.T

            C_up = np.column_stack([c11, c12])
            C_down = np.column_stack([c21, c22])
            C = np.row_stack([C_up, C_down])

            # print("c11:", c11)
            # print("c22:", c22)

            v1, Q1 = np.linalg.eig(c11)
            v2, Q2 = np.linalg.eig(c22)
            V1 = np.diag(v1 ** (-0.5))
            V2 = np.diag(v2 ** (-0.5))

            C11 = np.dot(np.dot(Q1, V1.T), np.linalg.inv(Q1))
            C22 = np.dot(np.dot(Q2, V2.T), np.linalg.inv(Q2))

            # print("Q1 * Q1^(-1):", np.dot(Q1, np.linalg.inv(Q1)))
            # print("Q2 * Q2^(-1):", np.dot(Q2, np.linalg.inv(Q2)))

            U_up = np.column_stack([C11, np.zeros((self.Nc, num_harm))])
            U_down = np.column_stack([np.zeros((y.shape[0], self.Nc)), C22])
            U = np.row_stack([U_up, U_down])
            R = np.dot(np.dot(U, C), U.T)

            eig_val, _ = np.linalg.eig(R)
            # print("eig_val:", eig_val, eig_val.shape)
            E = eig_val / np.sum(eig_val)
            S = 1 + np.sum(E * np.log(E)) / np.log(self.Nc + num_harm)
            result[freq_idx] = S

        return result

    def msi_classify(self, targets, test_data, num_harmonics=3):

        reference_signals = self.get_Reference_Signal(num_harmonics, targets)

        print("segmented_data.shape:", test_data.shape)
        print("reference_signals.shape:", reference_signals.shape)

        predicted_class = []
        labels = []
        num_segments = test_data.shape[0]
        num_perCls = num_segments // reference_signals.shape[0]

        for segment in range(0, num_segments):
            labels.append(segment // num_perCls)
            result = self.find_Synchronization_Index(test_data[segment], reference_signals)
            predicted_class.append(np.argmax(result) + 1)

        labels = np.array(labels) + 1
        predicted_class = np.array(predicted_class)
        return labels, predicted_class
