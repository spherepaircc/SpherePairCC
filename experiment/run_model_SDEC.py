import sys
sys.path.append("..")
sys.path.append('./')
import os
import csv
import torch.utils.data
import torch
import numpy as np
import argparse
import lib.loadData
from lib.loadData import *
from model.model_SDEC import SDEC
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='baseline: SDEC')
    parser.add_argument('--lr', type=float, default=0.01)                    # 0.01 for SGD, 0.001 for Adam
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0.001)
    parser.add_argument('--use_pretrain', type=str, default="True")          # Specify whether to use a pre-trained sdae model
    parser.add_argument('--use_kmeans', type=str, default="True")            # Specify whether to use kmeans to initialize cluster centers
    parser.add_argument('--lam', type=float, default=0.00001)                # Specify the pairwise loss weight lambda
    parser.add_argument('--dim', type=int, default=10)                       # Specify the dimension of the embedding space
    parser.add_argument('--expName', type=str, default="anchor_issues")      # Specify the experiment name, used to determine the path to store experimental results
    args = parser.parse_args()



    # ======================Load dataset=======================
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
    
    # Convert data to tensors, move to GPU, and get input dimension, number of clusters, embedding dimension
    X, test_X = [t.float().to(device) for t in (X, test_X)]
    y, test_y = [t.to(device) for t in (y, test_y)]
    input_dim = np.prod(X.shape[1:])
    n_clusters = len(np.unique(y.cpu().numpy()))
    z_dim = args.dim      # Dimension of the embedding space



    # ====================Create SDEC model====================
    model = SDEC(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    print(model)  # Print Network Structure



    # ======================Parameter settings======================
    # Decide whether to use a pre-trained sdae model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        pretrain_path = f'./pretrained_sdae_models/D{args.dim}/{args.dataset}_aeweights.pt'
        model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create folders to store various experiment results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_SDEC_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_SDEC_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create a folder to store record_log (training process logs)
    record_log_path = lab_result_path      # Same as the experiment results folder
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create a path to store encoder embedded features
    if args.expName == "tSNE":
        record_feature_path = lab_result_path   # Store in the same folder as the experiment results
        record_feature_dir = os.path.join(record_feature_path, f"feature_{args.dataset}_{args.consRule}_{args.consIndex}.pt")
    else:
        record_feature_dir = None

    # use_kmeans
    if args.use_kmeans == "True":
        use_kmeans = True
    else:
        use_kmeans = False



    # =====================Read constraints===================
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_SDEC_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_SDEC_noPretrain/savedCons/{args.dataset}_{args.consRule}'
    constraints_file = f'{cons_path}/constraints_{args.consIndex}.npz'
    constraints = np.load(constraints_file, allow_pickle=True)
    ml = constraints['ml']
    cl = constraints['cl']
    ml_ind1, ml_ind2 = zip(*ml) if ml.size > 0 else ([], [])
    cl_ind1, cl_ind2 = zip(*cl) if cl.size > 0 else ([], [])
    ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
    cl_ind1, cl_ind2 = np.array(cl_ind1), np.array(cl_ind2)
    # Shuffle the order of constraints
    ml_indices = np.arange(len(ml_ind1))
    np.random.shuffle(ml_indices)
    ml_ind1, ml_ind2 = ml_ind1[ml_indices], ml_ind2[ml_indices]
    cl_indices = np.arange(len(cl_ind1))
    np.random.shuffle(cl_indices)
    cl_ind1, cl_ind2 = cl_ind1[cl_indices], cl_ind2[cl_indices]



    # ====================Train the model and record results====================
    # Train the SDEC model
    epoch, train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari = model.fit(
        record_log_dir=record_log_dir,
        ml_ind1 = ml_ind1, 
        ml_ind2 = ml_ind2, 
        cl_ind1 = cl_ind1,
        cl_ind2 = cl_ind2,
        lam = args.lam,
        X = X,
        y = y, 
        test_X = test_X,
        test_y = test_y,
        lr = args.lr, 
        batch_size = args.batch_size, 
        num_epochs = args.epochs,
        tol = args.tol,
        use_kmeans = use_kmeans,
        record_feature_dir = record_feature_dir)



    # ======================Save final results======================
    if use_pretrain_flag == True:
        result_dir = os.path.join(lab_result_path, f"result_SDEC_Pretrain_{args.dataset}_{args.consRule}.csv")
    else:
        result_dir = os.path.join(lab_result_path, f"result_SDEC_noPretrain_{args.dataset}_{args.consRule}.csv")

    if not os.path.exists(result_dir):
        with open(result_dir, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["lr", "batch_size", 
                            "dataset", "consRule", "consIndex",
                            "epochs", 
                            "train_acc", "test_acc", 
                            "train_nmi", "test_nmi",
                            "train_ari", "test_ari"])
    with open(result_dir, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([args.lr, args.batch_size,
                        args.dataset, args.consRule, args.consIndex,
                        epoch, 
                        train_acc, test_acc, 
                        train_nmi, test_nmi,
                        train_ari, test_ari])
