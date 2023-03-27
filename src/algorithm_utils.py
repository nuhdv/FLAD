import abc
import logging
import random
from typing import List
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False, out_dir=None):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}
        self.out_dir = out_dir
        self.torch_save = False

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """

    def set_output_dir(self, out_dir):
        self.out_dir = out_dir

    def get_val_err(self):
        """
        :return: reconstruction error_tc for validation set,
        dimensions of num_val_time_points x num_channels
        Call after training
        """
        return None

    def get_val_loss(self):
        """
        :return: scalar loss after training
        """
        return None


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0
        self.torch_save = True

    @property
    def device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)

def get_sub_seqs(x_arr, seq_len, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs


def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits: List, seed: int,
                           shuffle: bool = False, usetorch=True):
    """
    Splits the train data between train, val, etc. Creates and returns pytorch data loaders
    :param shuffle: boolean that determines whether samples are shuffled before splitting the data
    :param seed: seed used for the random shuffling (if shuffling there is)
    :param x_seqs: input data where each row is a sample (a sequence) and each column is a channel
    :param batch_size: number of samples per batch
    :param splits: list of split fractions, should sum up to 1.
    :param usetorch: if True returns dataloaders, otherwise return datasets
    :return: a tuple of data loaders as long as splits. If len_splits = 1, only 1 data loader is returned
    """
    if np.sum(splits) != 1:
        scale_factor = np.sum(splits)
        splits = [fraction/scale_factor for fraction in splits]
    if shuffle:
        np.random.seed(seed)
        x_seqs = x_seqs[np.random.permutation(len(x_seqs))]
        np.random.seed()
    split_points = [0]
    for i in range(len(splits)-1):
        split_points.append(split_points[-1] + int(splits[i]*len(x_seqs)))
    split_points.append(len(x_seqs))
    if usetorch:
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]], batch_size=batch_size,
                                    drop_last=False, pin_memory=True, shuffle=False) for i in range(len(splits))])
        return loaders
    else:
        datasets = tuple([x_seqs[split_points[i]:split_points[i+1]] for i in range(len(splits))])
        return datasets