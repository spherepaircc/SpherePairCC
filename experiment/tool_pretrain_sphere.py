import sys
sys.path.append("..")
sys.path.append('./')
import os
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import argparse
from lib.stackedDAESphere import StackedDAE
from lib.datasets import *
import lib.loadData
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pretrain sdae for SpherePair')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pretrainepochs', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--dim', type=int, default=10)
    # Specify the network architecture, 
    # "normal" for standard network (500-500-2000), 
    # "compact" for compact network (256-256-512), 
    # "deep" for deep network (500-500-500-2000)
    parser.add_argument('--network', type=str, default="normal")
    args = parser.parse_args()


    #====================================Load Dataset====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')
    match = sub_pattern.match(dataset_name)

    if match:   # If it's a 'sub' dataset
        base_dataset = match.group('base')  # Base dataset, such as 'fmnist'
        n_class = int(match.group('n_class'))  # Number of subset classes
        load_function_name = f'load_{base_dataset}_sub'
        if hasattr(lib.loadData, load_function_name):
            load_function = getattr(lib.loadData, load_function_name)
            X, y, test_X, test_y = load_function(n_class)
        else:
            raise ValueError(f"Dataset loader for '{load_function_name}' not found.")
    else:       # If it's not a 'sub' dataset
        load_function_name = f'load_{dataset_name}'
        if hasattr(lib.loadData, load_function_name):
            load_function = getattr(lib.loadData, load_function_name)
            X, y, test_X, test_y, *_ = load_function()
        else:
            raise ValueError(f"Dataset loader for '{dataset_name}' not found.")
    
    X, test_X = [t.float().to(device) for t in (X, test_X)]
    y, test_y = [t.to(device) for t in (y, test_y)]
    input_dim = np.prod(X.shape[1:])
    z_dim = args.dim

    train_dataset = TensorDataset(X, y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    #======================Train Stacked Denoising Autoencoder and Save as Pretrained Model=======================
    sdae = StackedDAE(input_dim=input_dim, z_dim=z_dim, binary=False,
                      encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                      dropout=0)
    print(sdae)  # Print the structure of the pretrained SDAE model

    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
                 num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="mse")

    # Train the stacked denoising autoencoder
    sdae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs, corrupt=0.2, loss_type="mse")

    # sdae_model_save_path is determined by args.dim
    sdae_model_save_path = f"./pretrained_sphere_models/D{args.dim}"
    if not os.path.exists(sdae_model_save_path):
        os.makedirs(sdae_model_save_path)
    sdae_model_dir = os.path.join(sdae_model_save_path, f"{args.dataset}_aeweights.pt")
    sdae.save_model(sdae_model_dir)
