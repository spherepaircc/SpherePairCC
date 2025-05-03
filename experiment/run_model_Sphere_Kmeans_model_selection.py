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
    parser = argparse.ArgumentParser(description='Ours: SpherePairs + Kmeans')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--soft_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0)                 
    parser.add_argument('--use_pretrain', type=str, default="True")         # Specify whether to use a pre-trained sdae model
    parser.add_argument('--lam', type=float, default=0.02)                  # Specify the weight lambda value for reconstruction loss
    parser.add_argument('--omega', type=float, default=2)                   # Specify the omega value in pairwise loss
    parser.add_argument('--dim', type=int, default=10)                      # Specify the dimension of the embedding space
    parser.add_argument('--plot3D', type=str, default="False")              # Specify whether to plot a 3D spherical graph
    parser.add_argument('--expName', type=str, default="anchor_issues")     # Specify the experiment name, used to determine the path to store experiment results
    parser.add_argument('--autoK', type=str, default="False")               # Specify whether to use the method for automatically determining K value
    # Specify the network structure, 
    # "normal" indicates a standard network (500-500-2000), 
    # "compact" indicates a compact network (256-256-512), 
    # "deep" indicates a deep network (500-500-500-2000)
    parser.add_argument('--network', type=str, default="normal")            
    args = parser.parse_args()


    #====================== Load Dataset =======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')
    match = sub_pattern.match(dataset_name)

    if match:   # If it is a 'sub' dataset
        base_dataset = match.group('base')  # Base dataset, such as 'fmnist'
        n_class = int(match.group('n_class'))  # Number of subset classes
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
    if args.plot3D == True:
        z_dim = 3   # Used for plotting a 3D spherical graph
        args.dim = 3
    else:
        z_dim = args.dim



    #==================== Create sdae Model ====================
    if args.network == "normal":
        model = SpherePairs(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    elif args.network == "compact":
        model = SpherePairs(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                    encodeLayer=[256, 256, 512], decodeLayer=[512, 256, 256], activation="relu", dropout=0)
    elif args.network == "deep":
        model = SpherePairs(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                    encodeLayer=[500, 500, 500, 2000], decodeLayer=[2000, 500, 500, 500], activation="relu", dropout=0)
    print(model)  # Print Network Structure



    #====================== Parameter Settings ======================
    # Determine whether to use a pre-trained sdae model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        if args.network == "normal":
            pretrain_path = f'./pretrained_sphere_models/D{args.dim}/{args.dataset}_aeweights.pt'
            model.load_model(pretrain_path)
        elif args.network == "compact":
            pretrain_path = f'./pretrained_sphere_models_compact/D{args.dim}/{args.dataset}_aeweights.pt'
            model.load_model(pretrain_path)
        elif args.network == "deep":
            pretrain_path = f'./pretrained_sphere_models_deep/D{args.dim}/{args.dataset}_aeweights.pt'
            model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create folders to store various experiment results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_Sphere_Kmeans_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_Sphere_Kmeans_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create folders to store record_log (training process logs)
    record_log_path = lab_result_path        # Same folder as experiment results
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create paths to store features after encoder embedding
    if args.expName == "tSNE":
        record_feature_path = lab_result_path   # Same folder as experiment results
        record_feature_dir = os.path.join(record_feature_path, f"feature_{args.dataset}_{args.consRule}_{args.consIndex}.pt")
    else:
        record_feature_dir = None

    # Determine whether to plot a 3D spherical graph
    if args.plot3D == "True":
        plot_3D_path = os.path.join(record_log_path, f"plot_3D/")
    else:
        plot_3D_path = None




    #==================== Read Constraint Set for Training ====================
    # Read constraint set
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_Sphere_Kmeans_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_Sphere_Kmeans_noPretrain/savedCons/{args.dataset}_{args.consRule}'
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
    

    # Do not use the method for automatically determining K value, assume K value in ground truth is known
    if args.autoK == "False":
        #========================= Stage 2: Provide Assignment =========================
        # Make assignments
        train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari = model.assign_Kmeans(
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
        #====================== Save Final Results ======================
        if use_pretrain_flag == True:
            result_dir = os.path.join(lab_result_path, f"result_Sphere_Kmeans_Pretrain_{args.dataset}_{args.consRule}.csv")
        else:
            result_dir = os.path.join(lab_result_path, f"result_Sphere_Kmeans_noPretrain_{args.dataset}_{args.consRule}.csv")
        if not os.path.exists(result_dir):  # If the file does not exist, create it and write the header
            with open(result_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["lr", "batch_size", 
                                "dataset", "consRule", "consIndex",
                                "epochs", 
                                "train_acc", "test_acc", 
                                "train_nmi", "test_nmi",
                                "train_ari", "test_ari",
                                "D", "lam", "omega",
                                "network"])
        with open(result_dir, "a") as csvfile:  # Append experiment results
            writer = csv.writer(csvfile)
            writer.writerow([args.lr, args.batch_size,
                            args.dataset, args.consRule, args.consIndex,
                            epoch, 
                            train_acc, test_acc, 
                            train_nmi, test_nmi,
                            train_ari, test_ari,
                            args.dim, args.lam, args.omega,
                            args.network])
    else:
        silhouette_score_list = []
        train_acc_list, test_acc_list, train_nmi_list, test_nmi_list, train_ari_list, test_ari_list = [], [], [], [], [], []
        #========================= Automatically Determine K =========================
        # Iterate specific_K from 2 to 16
        search_max = 16
        for specific_K in range(2, search_max):
            # Make assignments
            train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari, silhouette_score = model.assign_Kmeans(
                record_log_dir=record_log_dir,
                ml_ind1 = ml_ind1, 
                ml_ind2 = ml_ind2, 
                cl_ind1 = cl_ind1,
                cl_ind2 = cl_ind2,
                X = X,
                y = y, 
                test_X = test_X,
                test_y = test_y,
                plot_3D_path = plot_3D_path,
                specific_K = specific_K)
            # Save silhouette_score and experiment results
            silhouette_score_list.append(silhouette_score)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_nmi_list.append(train_nmi)
            test_nmi_list.append(test_nmi)
            train_ari_list.append(train_ari)
            test_ari_list.append(test_ari)
        # Select the best specific_K
        best_specific_K = np.argmax(silhouette_score_list) + 2
        # =============== Stage 2: Provide Assignment Using the Best specific_K ===============
        # Make assignments
        train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari, best_silhouette_score = model.assign_Kmeans(
            record_log_dir=record_log_dir,
            ml_ind1 = ml_ind1, 
            ml_ind2 = ml_ind2, 
            cl_ind1 = cl_ind1,
            cl_ind2 = cl_ind2,
            X = X,
            y = y, 
            test_X = test_X,
            test_y = test_y,
            plot_3D_path = plot_3D_path,
            specific_K = best_specific_K)
        #====================== Save the Best Results ======================
        if use_pretrain_flag == True:
            result_dir = os.path.join(lab_result_path, f"result_Sphere_Kmeans_Pretrain_{args.dataset}_{args.consRule}.csv")
        else:
            result_dir = os.path.join(lab_result_path, f"result_Sphere_Kmeans_noPretrain_{args.dataset}_{args.consRule}.csv")
        if not os.path.exists(result_dir):
            with open(result_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["lr", "batch_size", 
                                "dataset", "consRule", "consIndex",
                                "epochs", 
                                "train_acc", "test_acc", 
                                "train_nmi", "test_nmi",
                                "train_ari", "test_ari",
                                "D", "lam", "omega", 
                                "best_specific_K"])
        with open(result_dir, "a") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([args.lr, args.batch_size,
                            args.dataset, args.consRule, args.consIndex,
                            epoch, 
                            train_acc, test_acc, 
                            train_nmi, test_nmi,
                            train_ari, test_ari,
                            args.dim, args.lam, args.omega,
                            best_specific_K])
        # =================== Save Experiment Results for Each specific_K Separately ===================
        silhouette_score_dir = os.path.join(record_log_path, f"silhouette_score_{args.dataset}_{args.consRule}_{args.consIndex}.csv")
        if not os.path.exists(silhouette_score_dir):
            with open(silhouette_score_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["dataset", "consRule", "consIndex",
                                 "specific_K", "silhouette_score",
                                 "train_acc", "test_acc",
                                 "train_nmi", "test_nmi",
                                 "train_ari", "test_ari"])
        with open(silhouette_score_dir, "a") as csvfile: 
            writer = csv.writer(csvfile)
            for i in range(2, 16):
                writer.writerow([args.dataset, args.consRule, args.consIndex,
                                 i, silhouette_score_list[i-2],
                                 train_acc_list[i-2], test_acc_list[i-2],
                                 train_nmi_list[i-2], test_nmi_list[i-2],
                                 train_ari_list[i-2], test_ari_list[i-2]])
