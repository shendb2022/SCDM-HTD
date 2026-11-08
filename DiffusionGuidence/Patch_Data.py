import matplotlib.pyplot as plt
# import spectral
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from .Utils import standard
from .ts_generation import ts_generation


def cosin_similarity(x, y):
    assert x.shape[-1] == y.shape[-1]
    x_y = np.sum(np.multiply(x, y), axis=-1)
    x_norm = np.sqrt(np.sum(x ** 2, axis=-1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=-1))
    sim = x_y / (x_norm * y_norm + 1e-8)
    # sim = np.reshape(sim, [x.shape[0], 1, 1],order='F')
    sim = np.reshape(sim, [x.shape[0], 1], order='F')
    return np.float32(sim)


class Data(data.Dataset):
    def __init__(self, path):
        mat = sio.loadmat(path)
        data = mat['data']
        gt = mat['map']
        # data[data<0] = 0
        data = standard(data)

        h, w, b = data.shape
        self.band = b
        pixel_nums = h * w

        ## get the target spectrum
        try:
            d = mat['d']
            target_spectrum = d
        except:
            d = ts_generation(data, gt, 7)
            target_spectrum = d

        background_samples = np.reshape(data, [-1, b], order='F')
        #
        # ## randomly generate target samples by linear representation
        alphas = np.random.uniform(0, 0.4, pixel_nums)

        alphas = alphas[:, None]
        target_samples = alphas * background_samples + (1 - alphas) * target_spectrum.T

        self.target_samples = target_samples
        self.background_samples = background_samples

        self.nums = self.target_samples.shape[0]

        self.target_label = cosin_similarity(self.target_samples, target_spectrum.T)
        self.background_label = cosin_similarity(self.background_samples, target_spectrum.T)

    def __getitem__(self, index):
        positive_samples = self.target_samples[index]
        positive_labels = self.target_label[index]
        # positive_samples = self.target_spectrum
        negative_samples = self.background_samples[index]
        negative_labels = self.background_label[index]
        return (torch.from_numpy(positive_labels).float(), torch.from_numpy(positive_samples).float()), (
            torch.from_numpy(negative_labels).float(), torch.from_numpy(negative_samples).float())

    def __len__(self):
        return self.nums


### This is for checking
if __name__ == '__main__':
    data = Data('/home/sdb/HTD-Mamba-3090/datasets2/Sandiego.mat')
    random_ind = np.random.randint(0, 10000, 1)
    target_couple, background_couple = data.__getitem__(random_ind)
    # print(background_couple[0].shape)
    # print(target_couple[1].shape)
    # target_samples = data.target_spectrum
    plt.plot(background_couple[0].T, label='0', color='r')
    plt.plot(background_couple[1].T, label='B', color='g')
    plt.plot(target_couple[0].T, label='t', color='b')
    plt.plot(target_couple[1].T, label='T', color='k')
    plt.legend()
    # # target_samples = data.target_spectrum
    # # print(target_samples.shape)
    # # plt.plot(target_samples.T)
    plt.show()

    ## Correct
