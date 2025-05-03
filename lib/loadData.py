import sys
import os
sys.path.append("..")
sys.path.append('./')
import torch.utils.data
import torch
from lib.datasets import MNIST, FashionMNIST, Reuters
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import numpy as np

def split_val_set(X, y, val_size=1000):
    X_val = X[:val_size]
    y_val = y[:val_size]
    X_new = X[val_size:]
    y_new = y[val_size:]
    return X_new, y_new, X_val, y_val

"""MNIST"""
def load_mnist(split_val=False):
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X_train = mnist_train.train_data
    y_train = mnist_train.train_labels
    X_test = mnist_test.test_data
    y_test = mnist_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test

"""MNIST -- sub classes"""
def load_mnist_sub(n_class, split_val=False):
    if not isinstance(n_class, int):
        raise TypeError("n_class must be an integer.")
    if not (1 <= n_class <= 10):
        raise ValueError("n_class must between 1 and 10.")

    selected_classes = list(range(n_class))  # the classes to be preserved

    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X_train = mnist_train.train_data
    y_train = mnist_train.train_labels
    X_test = mnist_test.test_data
    y_test = mnist_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    # filter out the selected classes
    train_mask = torch.zeros_like(y_train, dtype=torch.bool)
    test_mask = torch.zeros_like(y_test, dtype=torch.bool)
    for cls in selected_classes:
        train_mask |= (y_train == cls)
        test_mask |= (y_test == cls)
    X_train_sub = X_train[train_mask]
    y_train_sub = y_train[train_mask]
    X_test_sub = X_test[test_mask]
    y_test_sub = y_test[test_mask]

    if split_val:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub, X_val, y_val
    else:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub

"""FashionMNIST"""
def load_fmnist(split_val=False):
    fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
    fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
    X_train = fashionmnist_train.train_data
    y_train = fashionmnist_train.train_labels
    X_test = fashionmnist_test.test_data
    y_test = fashionmnist_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test

"""FashionMNIST -- raw data -- sub classes"""
def load_fmnist_sub(n_class, split_val=False):
    if not isinstance(n_class, int):
        raise TypeError("n_class must be an integer.")
    if not (1 <= n_class <= 10):
        raise ValueError("n_class must between 1 and 10.")

    selected_classes = list(range(n_class))
    fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
    fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
    X_train = fashionmnist_train.train_data
    y_train = fashionmnist_train.train_labels
    X_test = fashionmnist_test.test_data
    y_test = fashionmnist_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    train_mask = torch.zeros_like(y_train, dtype=torch.bool)
    test_mask = torch.zeros_like(y_test, dtype=torch.bool)
    for cls in selected_classes:
        train_mask |= (y_train == cls)
        test_mask |= (y_test == cls)
    X_train_sub = X_train[train_mask]
    y_train_sub = y_train[train_mask]
    X_test_sub = X_test[test_mask]
    y_test_sub = y_test[test_mask]

    if split_val:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub, X_val, y_val
    else:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub

"""Reuters -- raw data"""
def load_reuters(split_val=False):
    reuters_train = Reuters('./dataset/reuters', train=True, download=False)
    reuters_test = Reuters('./dataset/reuters', train=False)
    X_train = reuters_train.train_data
    y_train = reuters_train.train_labels
    X_test = reuters_test.test_data
    y_test = reuters_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test

"""Reuters -- raw data -- sub classes"""
def load_reuters_sub(n_class, split_val=False):
    if not isinstance(n_class, int):
        raise TypeError("n_class must be an integer.")
    if not (1 <= n_class <= 4):
        raise ValueError("n_class must between 1 and 4.")

    selected_classes = list(range(n_class))
    reuters_train = Reuters('./dataset/reuters', train=True, download=False)
    reuters_test = Reuters('./dataset/reuters', train=False)
    X_train = reuters_train.train_data
    y_train = reuters_train.train_labels
    X_test = reuters_test.test_data
    y_test = reuters_test.test_labels

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    train_mask = torch.zeros_like(y_train, dtype=torch.bool)
    test_mask = torch.zeros_like(y_test, dtype=torch.bool)
    for cls in selected_classes:
        train_mask |= (y_train == cls)
        test_mask |= (y_test == cls)
    X_train_sub = X_train[train_mask]
    y_train_sub = y_train[train_mask]
    X_test_sub = X_test[test_mask]
    y_test_sub = y_test[test_mask]

    if split_val:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub, X_val, y_val
    else:
        return X_train_sub, y_train_sub, X_test_sub, y_test_sub

"""STL10 -- pretrained/1000epoch/Resnet34/512dim"""
def load_stl10(split_val=False):
    pretrained_dataset_path = "./dataset/stl10/stl10_pretrained_dataset.pkl"
    with open(pretrained_dataset_path, 'rb') as f:
        X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if split_val:
        X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda')
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to('cuda')
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda')

    if split_val:
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to('cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to('cuda')
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor
    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

"""CIFAR10 -- pretrained/1000epoch/Resnet34/512dim"""
def load_cifar10(split_val=False):
    pretrained_dataset_path = "./dataset/cifar10/cifar10_pretrained_dataset.pkl"
    with open(pretrained_dataset_path, 'rb') as f:
        X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if split_val:
        X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda')
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to('cuda')
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda')

    if split_val:
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to('cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to('cuda')
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor
    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

"""CIFAR100-20 -- pretrained/1000epoch/Resnet34/512dim"""
def load_cifar100(split_val=False):
    pretrained_dataset_path = "./dataset/cifar100/cifar100_pretrained_dataset.pkl"
    with open(pretrained_dataset_path, 'rb') as f:
        X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if split_val:
        X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda')
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to('cuda')
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda')

    if split_val:
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to('cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to('cuda')
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor
    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

"""ImageNet10 -- pretrained/1000epoch/Resnet34/512dim"""
def load_imagenet10(split_val=False):
    pretrained_dataset_path = "./dataset/imagenet10/imagenet10_pretrained_dataset.pkl"
    with open(pretrained_dataset_path, 'rb') as f:
        X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if split_val:
        X_train, y_train, X_val, y_val = split_val_set(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if split_val:
        X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda')
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to('cuda')
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda')

    if split_val:
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to('cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to('cuda')
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor
    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
