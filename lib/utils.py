import os
import sys
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.linalg import norm
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



# Calculate the difference between two sets of labelings
def delta_label_sum(y1, y2):
    """
    Calculate the difference between two labelings. 
    Considering the different permutations of labels, we first need to find the best match/mapping, and then calculate the difference.
    """
    y1 = y1.astype(np.int64)
    y2 = y2.astype(np.int64)

    D = max(y1.max(), y2.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y1.size):
        w[y1[i], y2[i]] += 1

    # Find the best match
    from scipy.optimize import linear_sum_assignment as linear_assignment
    row_ind, col_ind = linear_assignment(w.max() - w)

    # Generate a mapping from one category to another
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Generate the mapped category labels
    y1_mapped = np.array([mapping[y] for y in y1])

    # Calculate the total number of mismatched labels
    delta = np.sum(y1_mapped != y2)

    return delta


# Get the best mapped representation of y to y_pred
def get_best_mapped_y(y, y_pred):
    """
    Get the best mapped representation of y to y_pred
    """
    y = y.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y.size):
        w[y[i], y_pred[i]] += 1

    # Find the best match
    from scipy.optimize import linear_sum_assignment as linear_assignment
    row_ind, col_ind = linear_assignment(w.max() - w)

    # Generate a mapping from one category to another
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Generate the mapped category labels
    y_mapped = np.array([mapping[y] for y in y])

    return y_mapped    # The return value is a numpy array



