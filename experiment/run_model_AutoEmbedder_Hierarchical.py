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
from model.model_AutoEmbedder import AutoEmbedder
import re


# autoembedder needs tuning of alpha
def get_best_alpha(finetune_file, dataset):
    """Get the best alpha for the specified dataset based on val_acc."""
    with open(finetune_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)

    dataset_idx = header.index(dataset)
    best_alpha = None
    best_acc = float('-inf')

    for row in data:
        alpha = float(row[0])
        val_acc = row[dataset_idx]

        if val_acc.startswith('(') and val_acc.endswith(')'):
            val_acc = float(eval(val_acc)[0])
        else:
            val_acc = float(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_alpha = alpha

    return best_alpha



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline: AutoEmbedder + Hierarchical')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--soft_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0)                  
    parser.add_argument('--use_pretrain', type=str, default="True")         # Specify whether to use a pretrained sdae model
    parser.add_argument('--alpha', type=float, default=100)                 # Euclidean distance margin for CL
    parser.add_argument('--dim', type=int, default=10)                      # Specify the dimension of the embedding space
    parser.add_argument('--expName', type=str, default="anchor_issues")     # Specify the experiment name for determining the path to store experimental results
    args = parser.parse_args()



    #====================== Load dataset ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')   # Regular expression to match 'dataset_sub_n'
    match = sub_pattern.match(dataset_name)

    if match:   # If it is a 'sub' dataset
        base_dataset = match.group('base')  # Base dataset, e.g., 'fmnist'
        n_class = int(match.group('n_class'))  # Number of classes in the subset
        load_function_name = f'load_{base_dataset}_sub'
        if hasattr(lib.loadData, load_function_name):
            load_function = getattr(lib.loadData, load_function_name)
            X, y, test_X, test_y = load_function(n_class)
        else:
            raise ValueError(f"Dataset loader for '{load_function_name}' not found.")
    else:       # If it is not a 'sub' dataset
        load_function_name = f'load_{dataset_name}'
        if hasattr(lib.loadData, load_function_name):
            load_function = getattr(lib.loadData, load_function_name)
            X, y, test_X, test_y, *_ = load_function()
        else:
            raise ValueError(f"Dataset loader for '{dataset_name}' not found.")
    
    X, test_X = [t.float().to(device) for t in (X, test_X)]
    y, test_y = [t.to(device) for t in (y, test_y)]
    input_dim = np.prod(X.shape[1:])
    n_clusters = len(np.unique(y.cpu().numpy()))
    z_dim = args.dim



    #==================== Create sdae model ====================
    # Load the record file of finetune results
    finetune_file = f"./exp_finetune/finetune_AutoEmbedder.csv"

    # CL margin of AutoEmbedder
    # Use the alpha value corresponding to the current dataset in the recorded file
    # Read the corresponding alpha value from the file
    if match:
        this_dataset = base_dataset
    else:
        this_dataset = args.dataset
    alpha = get_best_alpha(finetune_file, this_dataset)
    print("Best alpha loaded:", alpha)

    model = AutoEmbedder(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", alpha=alpha)
    print(model)  # Print Network Structure



    #====================== Parameter settings ======================
    # Determine whether to use a pretrained sdae model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        pretrain_path = f'./pretrained_sdae_models/D{args.dim}/{args.dataset}_aeweights.pt'
        model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create a folder to store various experimental results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_AutoEmbedder_Hierarchical_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_AutoEmbedder_Hierarchical_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create a folder to store record logs (training process logs)
    record_log_path = lab_result_path        # Store in the same folder as experimental results
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create a path to store features after encoder embedding
    if args.expName == "tSNE":
        record_feature_path = lab_result_path   # Store in the same folder as experimental results
        record_feature_dir = os.path.join(record_feature_path, f"feature_{args.dataset}_{args.consRule}_{args.consIndex}.pt")
    else:
        record_feature_dir = None

    

    #==================== Read constraint set ===================
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_AutoEmbedder_Hierarchical_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_AutoEmbedder_Hierarchical_noPretrain/savedCons/{args.dataset}_{args.consRule}'
    constraints_file = f'{cons_path}/constraints_{args.consIndex}.npz'
    constraints = np.load(constraints_file, allow_pickle=True)
    ml = constraints['ml']
    cl = constraints['cl']
    ml_ind1, ml_ind2 = zip(*ml) if ml.size > 0 else ([], [])
    cl_ind1, cl_ind2 = zip(*cl) if cl.size > 0 else ([], [])
    ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
    cl_ind1, cl_ind2 = np.array(cl_ind1), np.array(cl_ind2)
    # shuffle the constraints
    ml_indices = np.arange(len(ml_ind1))
    np.random.shuffle(ml_indices)
    ml_ind1, ml_ind2 = ml_ind1[ml_indices], ml_ind2[ml_indices]
    cl_indices = np.arange(len(cl_ind1))
    np.random.shuffle(cl_indices)
    cl_ind1, cl_ind2 = cl_ind1[cl_indices], cl_ind2[cl_indices]



    #==================== Stage 1: Train model embedding ====================
    # Train the model for AutoEmbedder embedding
    epoch = model.fit(
        record_log_dir = record_log_dir,
        ml_ind1 = ml_ind1, 
        ml_ind2 = ml_ind2, 
        cl_ind1 = cl_ind1,
        cl_ind2 = cl_ind2,
        X = X,
        y = y, 
        lr = args.lr, 
        batch_size = args.batch_size, 
        epochs = args.epochs,
        soft_epochs = args.soft_epochs,
        tol = args.tol,
        record_feature_dir = record_feature_dir)
    

    #========================= Stage 2: Make assignment =========================
    # Make assignments
    train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari = model.assign_Hierarchical(
        record_log_dir=record_log_dir,
        ml_ind1 = ml_ind1, 
        ml_ind2 = ml_ind2, 
        cl_ind1 = cl_ind1,
        cl_ind2 = cl_ind2,
        X = X,
        y = y, 
        test_X = test_X,
        test_y = test_y)
    
    #====================== Save final results ======================
    if use_pretrain_flag == True:
        result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Hierarchical_Pretrain_{args.dataset}_{args.consRule}.csv")
    else:
        result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Hierarchical_noPretrain_{args.dataset}_{args.consRule}.csv")
    if not os.path.exists(result_dir):
        with open(result_dir, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["lr", "batch_size", 
                            "dataset", "consRule", "consIndex",
                            "epochs", 
                            "train_acc", "test_acc", 
                            "train_nmi", "test_nmi",
                            "train_ari", "test_ari",
                            "D", "alpha"])
    with open(result_dir, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([args.lr, args.batch_size,
                        args.dataset, args.consRule, args.consIndex,
                        epoch, 
                        train_acc, test_acc, 
                        train_nmi, test_nmi,
                        train_ari, test_ari,
                        args.dim, alpha])
        
