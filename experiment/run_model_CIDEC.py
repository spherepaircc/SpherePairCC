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
from model.model_CIDEC import CIDEC
import re
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='baseline: C-IDEC')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0.001)
    parser.add_argument('--valCons_on_testset', type=str, default="False")    # Use constraints on the test set to observe model behavior
    parser.add_argument('--use_pretrain', type=str, default="True")           # Specify whether to use a pretrained SDAE model
    parser.add_argument('--use_kmeans', type=str, default="True")             # Specify whether to use k-means to initialize cluster centers
    parser.add_argument('--ml_penalty', type=float, default=0.1)              # Specify the penalty coefficient for ML constraints
    parser.add_argument('--cl_penalty', type=float, default=1)                # Specify the penalty coefficient for CL constraints
    parser.add_argument('--clustering_loss_weight', type=float, default=1)    # Specify the trade-off between unsupervised clustering loss and reconstruction loss
    parser.add_argument('--dim', type=int, default=10)                        # Specify the dimension of the embedding space
    parser.add_argument('--expName', type=str, default="anchor_issues")       # Specify the experiment name, used to determine the path for storing experiment results
    args = parser.parse_args()



    #======================Load Dataset=======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')
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
    z_dim = args.dim      # Dimension of the embedding space



    #====================Create SDAE Model====================
    model = CIDEC(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    print(model)  # Print Network Structure



    #======================Parameter Settings======================
    # Determine whether to use a pretrained SDAE model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        pretrain_path = f'./pretrained_sdae_models/D{args.dim}/{args.dataset}_aeweights.pt'
        model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create folders to store various experiment results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_CIDEC_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_CIDEC_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create folder to store record_log (training process logs)
    record_log_path = lab_result_path      # Same as the experiment results folder
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create path to store encoder-embedded features
    if args.expName == "tSNE":
        record_feature_path = lab_result_path  # Store in the same folder as experiment results
        prefix = f"feature_{args.dataset}_{args.consRule}_{args.consIndex}"
        pattern = os.path.join(record_feature_path, prefix + "*.pt")
        matching_files = glob.glob(pattern)

        if not matching_files:
            new_file_name = prefix + "_00.pt"
        else:
            max_num = -1
            for file_path in matching_files:
                base_name = os.path.basename(file_path)
                parts = base_name.split('_')
                if len(parts) >= 5 and parts[-1].endswith(".pt"):
                    num_part = parts[-1][:-3]
                    try:
                        num = int(num_part)
                        if num > max_num:
                            max_num = num
                    except ValueError:
                        continue
            new_num = max_num + 1
            new_file_name = prefix + f"_{new_num:02d}.pt"
        record_feature_dir = os.path.join(record_feature_path, new_file_name)
    else:
        record_feature_dir = None

    # use_kmeans
    if args.use_kmeans == "True":
        use_kmeans = True
    else:
        use_kmeans = False




    #====================Load Constraints for Training Set===================
    # Load constraints
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_CIDEC_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_CIDEC_noPretrain/savedCons/{args.dataset}_{args.consRule}'
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



    #=====================Load Constraints for Test Set to Observe====================
    if args.valCons_on_testset == "True":
        # Load constraints on the test set for observation
        if use_pretrain_flag == True:
            cons_path_test = f'./exp_{args.expName}/lab_CIDEC_Pretrain/savedCons_test/{args.dataset}_balance'
        else:
            cons_path_test = f'./exp_{args.expName}/lab_CIDEC_noPretrain/savedCons_test/{args.dataset}_balance'
        constraints_file_test = f'{cons_path_test}/constraints_{args.consIndex}.npz'
        constraints_test = np.load(constraints_file_test, allow_pickle=True)
        ml_test = constraints_test['ml']
        cl_test = constraints_test['cl']
        ml_ind1_test, ml_ind2_test = zip(*ml_test) if ml_test.size > 0 else ([], [])
        cl_ind1_test, cl_ind2_test = zip(*cl_test) if cl_test.size > 0 else ([], [])
        ml_ind1_test, ml_ind2_test = np.array(ml_ind1_test), np.array(ml_ind2_test)
        cl_ind1_test, cl_ind2_test = np.array(cl_ind1_test), np.array(cl_ind2_test)
        # Shuffle the order of constraints
        ml_indices_test = np.arange(len(ml_ind1_test))
        np.random.shuffle(ml_indices_test)
        ml_ind1_test, ml_ind2_test = ml_ind1_test[ml_indices_test], ml_ind2_test[ml_indices_test]
        cl_indices_test = np.arange(len(cl_ind1_test))
        np.random.shuffle(cl_indices_test)
        cl_ind1_test, cl_ind2_test = cl_ind1_test[cl_indices_test], cl_ind2_test[cl_indices_test]
    else:
        ml_ind1_test, ml_ind2_test, cl_ind1_test, cl_ind2_test = None, None, None, None



    #====================Train Model and Record Results====================
    # Train the CIDEC model
    epoch, train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari = model.fit(
        record_log_dir=record_log_dir,
        ml_ind1 = ml_ind1, 
        ml_ind2 = ml_ind2, 
        cl_ind1 = cl_ind1,
        cl_ind2 = cl_ind2,
        val_ml_ind1 = ml_ind1_test, 
        val_ml_ind2 = ml_ind2_test, 
        val_cl_ind1 = cl_ind1_test, 
        val_cl_ind2 = cl_ind2_test,
        ml_p = args.ml_penalty, 
        cl_p = args.cl_penalty,
        X = X,
        y = y, 
        test_X = test_X,
        test_y = test_y,
        lr = args.lr, 
        batch_size = args.batch_size, 
        num_epochs = args.epochs,
        tol = args.tol,
        use_kmeans = use_kmeans,
        clustering_loss_weight = args.clustering_loss_weight,
        record_feature_dir = record_feature_dir)



    #======================Save Final Results======================
    if use_pretrain_flag == True:
        result_dir = os.path.join(lab_result_path, f"result_CIDEC_Pretrain_{args.dataset}_{args.consRule}.csv")
    else:
        result_dir = os.path.join(lab_result_path, f"result_CIDEC_noPretrain_{args.dataset}_{args.consRule}.csv")

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
