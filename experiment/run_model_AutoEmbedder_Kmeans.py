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
import glob


def initialize_finetune_file(finetune_file):
    """Initialize the finetune file with headers and default values."""
    if not os.path.exists(finetune_file):
        with open(finetune_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['alpha', 'mnist', 'fmnist', 'reuters', 'cifar10', 'stl10', 'imagenet10', 'cifar100', 'cifar100-D20', 'cifar100-D30'])
            # Initialize rows for alpha values
            for alpha in [1, 10, 50, 100, 500, 1000, 5000, 10000]:
                writer.writerow([alpha, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def update_finetune_file(finetune_file, dataset, alpha, val_acc):
    """Update the finetune file with val_acc for the specified dataset and alpha."""
    data = []
    with open(finetune_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)

    # Find the dataset column index and alpha row index
    dataset_idx = header.index(dataset)
    alpha_row_idx = next((i for i, row in enumerate(data) if float(row[0]) == alpha), None)

    if alpha_row_idx is not None:
        data[alpha_row_idx][dataset_idx] = val_acc  # Update val_acc

    with open(finetune_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


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

        # Compare and update the best alpha
        if val_acc > best_acc:
            best_acc = val_acc
            best_alpha = alpha

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline: AutoEmbedder + Kmeans')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--soft_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0)                  
    parser.add_argument('--use_pretrain', type=str, default="True")         # Specify whether to use a pre-trained SDAE model
    parser.add_argument('--alpha', type=float, default=100)                 # Euclidean distance margin
    parser.add_argument('--dim', type=int, default=10)                      # Specify the dimension of the embedding space
    parser.add_argument('--expName', type=str, default="anchor_issues")     # Specify the experiment name, used to determine the path to store experiment results
    parser.add_argument('--autoK', type=str, default="False")               # Specify whether to use the method for automatically determining the K value
    parser.add_argument('--finetune_alpha', type=float, default=None)       # When specified, it is the hyperparameter tuning phase; if not specified, it is a normal experiment
    args = parser.parse_args()



    #====================== Load Dataset =======================
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
            if args.finetune_alpha is not None:
                X, y, test_X, test_y, val_X, val_y = load_function(split_val=True)
            else:
                X, y, test_X, test_y = load_function()
        else:
            raise ValueError(f"Dataset loader for '{dataset_name}' not found.")
    
    if match:
        X, test_X = [t.float().to(device) for t in (X, test_X)]
        y, test_y = [t.to(device) for t in (y, test_y)]
    else:
        if args.finetune_alpha is not None:
            X, test_X, val_X = [t.float().to(device) for t in (X, test_X, val_X)]
            y, test_y, val_y = [t.to(device) for t in (y, test_y, val_y)]
        else:
            X, test_X = [t.float().to(device) for t in (X, test_X)]
            y, test_y = [t.to(device) for t in (y, test_y)]
    input_dim = np.prod(X.shape[1:])
    n_clusters = len(np.unique(y.cpu().numpy()))
    z_dim = args.dim



    #==================== Create SDAE Model ====================
    # Initialize or load finetune record file
    finetune_file = f"./exp_finetune/finetune_AutoEmbedder.csv"
    initialize_finetune_file(finetune_file)

    # CL margin for AutoEmbedder,
    # If finetune_alpha is specified, use the specified value for the validation set and record the results in a file
    # If finetune_alpha is not specified, use the alpha value corresponding to the current dataset from the recorded file
    if args.finetune_alpha is not None:   # When finetune_alpha is specified, directly use the specified alpha value for finetuning
        alpha = args.finetune_alpha
    else:  # When finetune_alpha is not specified, read the corresponding alpha value from the file
        if match:
            this_dataset = base_dataset
        else:
            this_dataset = args.dataset
        if this_dataset == "cifar100" and args.dim == 20:
            this_dataset = "cifar100-D20"
        elif this_dataset == "cifar100" and args.dim == 30:
            this_dataset = "cifar100-D30"
        alpha = get_best_alpha(finetune_file, this_dataset)
        print("Best alpha loaded:", alpha)
    
    model = AutoEmbedder(input_dim=input_dim, z_dim=z_dim, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", alpha=alpha)
    print(model)  # Print Network Structure



    #====================== Parameter Settings =======================
    # Decide whether to use a pre-trained SDAE model
    use_pretrain_flag = False
    if args.use_pretrain == "True":
        if args.finetune_alpha is not None:
            pretrain_path = f'./pretrained_sdae_models/D{args.dim}/finetune/{args.dataset}_aeweights.pt'
            # pretrain_path = f'./pretrained_sdae_models/D{args.dim}/{args.dataset}_aeweights.pt'
            print(f"Finetune with pretrain model: {pretrain_path}")
        else:
            pretrain_path = f'./pretrained_sdae_models/D{args.dim}/{args.dataset}_aeweights.pt'
        model.load_model(pretrain_path)
        use_pretrain_flag = True

    # Create folders to store various experiment results
    if use_pretrain_flag == True:
        lab_result_path = f"./exp_{args.expName}/lab_AutoEmbedder_Kmeans_Pretrain/{args.dataset}/{args.consRule}"
    else:
        lab_result_path = f"./exp_{args.expName}/lab_AutoEmbedder_Kmeans_noPretrain/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create a folder to store record_log (training process logs)
    record_log_path = lab_result_path        # In the same folder as experiment results
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Create a path to store encoder-embedded features
    if args.expName == "tSNE":
        record_feature_path = lab_result_path
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
    
    # Whether to record the changes in feature norm
    if args.expName == "feature_norm":
        record_feature_norm = True
    else:
        record_feature_norm = False



    #==================== Load Constraint Set ====================
    if use_pretrain_flag == True:
        cons_path = f'./exp_{args.expName}/lab_AutoEmbedder_Kmeans_Pretrain/savedCons/{args.dataset}_{args.consRule}'
    else:
        cons_path = f'./exp_{args.expName}/lab_AutoEmbedder_Kmeans_noPretrain/savedCons/{args.dataset}_{args.consRule}'
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



    #==================== Stage 1: Train Model Embedding ====================
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
        record_feature_dir = record_feature_dir,
        record_feature_norm = record_feature_norm)
    
    #======================= When finetune_alpha is specified, record val_acc for the current dataset using the current alpha on the validation set =======================
    if args.finetune_alpha is not None:   # When finetune_alpha is specified, record val_acc
        # Test the model on the validation set
        # Make assignments
        train_acc, val_acc, train_nmi, val_nmi, train_ari, val_ari = model.assign_Kmeans(
            record_log_dir=record_log_dir,
            ml_ind1 = ml_ind1, 
            ml_ind2 = ml_ind2, 
            cl_ind1 = cl_ind1,
            cl_ind2 = cl_ind2,
            X = X,
            y = y, 
            test_X = val_X,
            test_y = val_y)
        # Record dataset, alpha, val_acc to file
        record_dataset = args.dataset
        if record_dataset == "cifar100" and args.dim == 20:
            record_dataset = "cifar100-D20"
        elif record_dataset == "cifar100" and args.dim == 30:
            record_dataset = "cifar100-D30"
        update_finetune_file(finetune_file, record_dataset, args.finetune_alpha, val_acc)


    #======================= When finetune_alpha is not specified, proceed as usual =======================
    else:
        # Do not use the method for automatically determining K; directly use the ground truth K
        if args.autoK == "False":
            #========================= Stage 2: Make Assignments =========================
            import time
            start_time = time.time()
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
                test_y = test_y)
            end_time = time.time()
            cluster_time = end_time - start_time
            print(f"Time for clustering: {cluster_time:.2f}s")

            #====================== Save Final Results ======================
            if use_pretrain_flag == True:
                result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Kmeans_Pretrain_{args.dataset}_{args.consRule}.csv")
            else:
                result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Kmeans_noPretrain_{args.dataset}_{args.consRule}.csv")
            if not os.path.exists(result_dir):
                with open(result_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["lr", "batch_size", 
                                    "dataset", "consRule", "consIndex",
                                    "epochs", 
                                    "train_acc", "test_acc", 
                                    "train_nmi", "test_nmi",
                                    "train_ari", "test_ari",
                                    "D", "alpha", "cluster_time"])
            with open(result_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([args.lr, args.batch_size,
                                args.dataset, args.consRule, args.consIndex,
                                epoch, 
                                train_acc, test_acc, 
                                train_nmi, test_nmi,
                                train_ari, test_ari,
                                args.dim, alpha, cluster_time])
        else:
            silhouette_score_list = []
            silhouette_score_std_list = []
            train_acc_list, test_acc_list, train_nmi_list, test_nmi_list, train_ari_list, test_ari_list = [], [], [], [], [], []
            #========================= Automatically Determine K =========================
            # Iterate over specific_K
            specific_K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            for specific_K in specific_K_values:
                # Make assignments
                train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari, silhouette_score_avg, silhouette_score_std = model.assign_Kmeans(
                    record_log_dir=record_log_dir,
                    ml_ind1 = ml_ind1, 
                    ml_ind2 = ml_ind2, 
                    cl_ind1 = cl_ind1,
                    cl_ind2 = cl_ind2,
                    X = X,
                    y = y, 
                    test_X = test_X,
                    test_y = test_y,
                    specific_K = specific_K)
                # Save silhouette_score and experiment results
                silhouette_score_list.append(silhouette_score_avg)
                silhouette_score_std_list.append(silhouette_score_std)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                train_nmi_list.append(train_nmi)
                test_nmi_list.append(test_nmi)
                train_ari_list.append(train_ari)
                test_ari_list.append(test_ari)
            # Select the best specific_K
            best_specific_K = np.argmax(silhouette_score_list) + 2
            # =============== Stage 2: Make Assignments Using the Best specific_K ===============
            # Make assignments
            train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari, best_silhouette_score, best_silhouette_score_std = model.assign_Kmeans(
                record_log_dir=record_log_dir,
                ml_ind1 = ml_ind1, 
                ml_ind2 = ml_ind2, 
                cl_ind1 = cl_ind1,
                cl_ind2 = cl_ind2,
                X = X,
                y = y, 
                test_X = test_X,
                test_y = test_y,
                specific_K = best_specific_K)
            #====================== Save the Best Results ======================
            if use_pretrain_flag == True:
                result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Kmeans_Pretrain_{args.dataset}_{args.consRule}.csv")
            else:
                result_dir = os.path.join(lab_result_path, f"result_AutoEmbedder_Kmeans_noPretrain_{args.dataset}_{args.consRule}.csv")
            if not os.path.exists(result_dir):
                with open(result_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["lr", "batch_size", 
                                    "dataset", "consRule", "consIndex",
                                    "epochs", 
                                    "train_acc", "test_acc", 
                                    "train_nmi", "test_nmi",
                                    "train_ari", "test_ari",
                                    "D", "alpha", 
                                    "best_specific_K"])
            with open(result_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([args.lr, args.batch_size,
                                args.dataset, args.consRule, args.consIndex,
                                epoch, 
                                train_acc, test_acc, 
                                train_nmi, test_nmi,
                                train_ari, test_ari,
                                args.dim, alpha,
                                best_specific_K])
            # =================== Save Experiment Results for Each specific_K in a Separate File ===================
            silhouette_score_dir = os.path.join(record_log_path, f"silhouette_score_{args.dataset}_{args.consRule}_{args.consIndex}.csv")
            if not os.path.exists(silhouette_score_dir):
                with open(silhouette_score_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["dataset", "consRule", "consIndex",
                                    "specific_K", "silhouette_score", "silhouette_score_std",
                                    "train_acc", "test_acc",
                                    "train_nmi", "test_nmi",
                                    "train_ari", "test_ari"])
            with open(silhouette_score_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(specific_K_values)):
                    writer.writerow([args.dataset, args.consRule, args.consIndex,
                                    specific_K_values[i], silhouette_score_list[i], silhouette_score_std_list[i], 
                                    train_acc_list[i], test_acc_list[i],
                                    train_nmi_list[i], test_nmi_list[i],
                                    train_ari_list[i], test_ari_list[i]])
