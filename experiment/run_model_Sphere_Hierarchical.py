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
from model.model_SpherePairs import SpherePairs
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ours: SpherePairs + Hierarchical')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--soft_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0)
    parser.add_argument('--use_pretrain', type=str, default="True")         # Specify whether to use a pre-trained SDAE model
    parser.add_argument('--lam', type=float, default=0.02)                  # Specify the reconstruction loss weight lambda value
    parser.add_argument('--omega', type=float, default=2)                   # Specify the omega value in pairwise loss
    parser.add_argument('--dim', type=int, default=10)                      # Specify the dimension of the embedding space
    parser.add_argument('--plot3D', type=str, default="False")              # Specify whether to plot a 3D sphere graph
    parser.add_argument('--expName', type=str, default="anchor_issues")     # Specify the experiment name to determine the path for storing experiment results
    parser.add_argument('--autoK', type=str, default="False")               # Specify whether to use the method for automatically determining the K value
    args = parser.parse_args()



    #======================Load Dataset=======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')
    match = sub_pattern.match(dataset_name)

    if match:   # If it's a 'sub' dataset
        base_dataset = match.group('base')  # Base dataset, e.g., 'fmnist'
        n_class = int(match.group('n_class'))  # Number of subset classes
        load_function_name = f'load_{base_dataset}_sub'
        if hasattr(lib.loadData, load_function_name):
            load_function = getattr(lib.loadData, load_function_name)
            X, y, test_X, test_y = load_function(n_class)
        else:
            raise ValueError(f"Dataset loader for '{load_function_name}' not found.")
    else:       # If not a 'sub' dataset
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
    if args.plot3D == True:
        z_dim = 3   # Used to plot a 3D sphere graph
        args.dim = 3
    else:
        z_dim = args.dim


    #====================Create SpherePairs Model====================
    model = SpherePairs(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    print(model)  # Print Network Structure



    #======================Parameter Settings======================
    # Decide whether to use a pre-trained SDAE model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        pretrain_path = f'./pretrained_sphere_models/D{args.dim}/{args.dataset}_aeweights.pt'
        model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create folder to store various experiment results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_Sphere_Hierarchical_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_Sphere_Hierarchical_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create folder to store record_log (training process log)
    record_log_path = lab_result_path        # Same folder as experiment results
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create path to store features after encoder embedding
    if args.expName == "tSNE":
        record_feature_path = lab_result_path   # Same folder as experiment results
        record_feature_dir = os.path.join(record_feature_path, f"feature_{args.dataset}_{args.consRule}_{args.consIndex}.pt")
    else:
        record_feature_dir = None

    # Decide whether to plot a 3D sphere graph
    if args.plot3D == "True":
        plot_3D_path = os.path.join(record_log_path, f"plot_3D/")
    else:
        plot_3D_path = None



    #====================Load Constraint Set===================
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_Sphere_Hierarchical_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_Sphere_Hierarchical_noPretrain/savedCons/{args.dataset}_{args.consRule}'
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



    #==================== Stage 1: Train Model's Embedding ====================
    # Train the model for Sphere embedding
    epoch = model.fit(
        record_log_dir = record_log_dir,
        ml_ind1 = ml_ind1, 
        ml_ind2 = ml_ind2, 
        cl_ind1 = cl_ind1,
        cl_ind2 = cl_ind2,
        lam = args.lam,
        X = X,
        y = y, 
        lr = args.lr, 
        batch_size = args.batch_size, 
        epochs = args.epochs,
        soft_epochs = args.soft_epochs,
        tol = args.tol,
        omega = args.omega,
        plot_3D_path = plot_3D_path,
        record_feature_dir = record_feature_dir)
    

    #========================= Stage 2: Make Assignment =========================
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
        test_y = test_y,
        plot_3D_path = plot_3D_path)
    

    #======================Save Final Results======================
    if use_pretrain_flag == True:
        result_dir = os.path.join(lab_result_path, f"result_Sphere_Hierarchical_Pretrain_{args.dataset}_{args.consRule}.csv")
    else:
        result_dir = os.path.join(lab_result_path, f"result_Sphere_Hierarchical_noPretrain_{args.dataset}_{args.consRule}.csv")
    if not os.path.exists(result_dir):
        with open(result_dir, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["lr", "batch_size", 
                            "dataset", "consRule", "consIndex",
                            "epochs", 
                            "train_acc", "test_acc", 
                            "train_nmi", "test_nmi",
                            "train_ari", "test_ari",
                            "D", "lam", "omega"])
    with open(result_dir, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([args.lr, args.batch_size,
                        args.dataset, args.consRule, args.consIndex,
                        epoch, 
                        train_acc, test_acc, 
                        train_nmi, test_nmi,
                        train_ari, test_ari,
                        args.dim, args.lam, args.omega])
