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
from model.model_VolMaxDCC import VolMaxDCC
import re


# VolMaxDCC needs tuning of lambda
def initialize_finetune_file(finetune_file):
    """Initialize the finetune file with headers and default values."""
    if not os.path.exists(finetune_file):
        with open(finetune_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['lam', 'mnist', 'fmnist', 'reuters', 'cifar10', 'stl10', 'imagenet10', 'cifar100'])
            # Initialize rows for lambda values
            for lam in [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                writer.writerow([lam, 0, 0, 0, 0, 0, 0, 0])


def update_finetune_file(finetune_file, dataset, lam, val_acc):
    """Update the finetune file with val_acc for the specified dataset and lambda."""
    data = []
    with open(finetune_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)
    dataset_idx = header.index(dataset)
    lam_row_idx = next((i for i, row in enumerate(data) if float(row[0]) == lam), None)
    if lam_row_idx is not None:
        data[lam_row_idx][dataset_idx] = val_acc
    with open(finetune_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


def get_best_lambda(finetune_file, dataset):
    """Get the best lambda for the specified dataset based on val_acc."""
    with open(finetune_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)

    dataset_idx = header.index(dataset)
    best_lambda = None
    best_acc = float('-inf')

    for row in data:
        lam = float(row[0])
        val_acc = row[dataset_idx]

        if val_acc.startswith('(') and val_acc.endswith(')'):
            val_acc = float(eval(val_acc)[0])
        else:
            val_acc = float(val_acc)

        # Compare and update the best lambda
        if val_acc > best_acc:
            best_acc = val_acc
            best_lambda = lam

    return best_lambda



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='baseline: VolMaxDCC')
    parser.add_argument('--lr', type=float, default=0.5)                    # 0.5 for SGD, 0.001 for Adam
    parser.add_argument('--batch-size', type=int, default=128)              # 128 for SGD, 256 for Adam
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0.001)
    parser.add_argument('--valCons_on_testset', type=str, default="False")   # Use constraints on the test set to observe model behavior
    parser.add_argument('--expName', type=str, default="anchor_issues")     # Specify the experiment name, used to determine the path to store experiment results
    parser.add_argument('--finetune_lambda', type=float, default=None)      # When specified, it is the tuning phase; if not specified, it is a normal experiment
    args = parser.parse_args()



    #======================Load Dataset=======================
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
            if args.finetune_lambda is not None:
                X, y, test_X, test_y, val_X, val_y = load_function(split_val=True)
            else:
                X, y, test_X, test_y = load_function()
        else:
            raise ValueError(f"Dataset loader for '{dataset_name}' not found.")
    
    if args.finetune_lambda is not None:
        X, test_X, val_X = [t.float().to(device) for t in (X, test_X, val_X)]
        y, test_y, val_y = [t.to(device) for t in (y, test_y, val_y)]
    else:
        X, test_X = [t.float().to(device) for t in (X, test_X)]
        y, test_y = [t.to(device) for t in (y, test_y)]
    input_dim = np.prod(X.shape[1:])
    n_clusters = len(np.unique(y.cpu().numpy()))



    #====================Create Model====================
    model = VolMaxDCC(input_dim=input_dim, n_clusters=n_clusters,
                hidden_layers=[512, 512], activation="relu")
    print(model)  # Print Network Structure



    #======================Parameter Setup======================
    # Create a folder to store various experiment results
    lab_result_path = f"./exp_{args.expName}/lab_VolMaxDCC/{args.dataset}/{args.consRule}"
    if not os.path.exists(lab_result_path):
        os.makedirs(lab_result_path)
    
    # Create a folder to store the record_log (training process logs)
    record_log_path = lab_result_path        # Same folder as the experiment results
    if not os.path.exists(record_log_path):
        os.makedirs(record_log_path)
    record_log_dir = os.path.join(record_log_path, f"log_{args.dataset}_{args.consRule}_{args.consIndex}.csv") 

    # Initialize or load the finetune record file
    finetune_file = f"./exp_finetune/finetune_VolMaxDCC.csv"
    initialize_finetune_file(finetune_file)

    # Trade-off between the two losses of VolMaxDCC
    # If finetune_lambda is specified, use the specified value for the validation set and record the results in a file
    # If finetune_lambda is not specified, use the lambda value corresponding to the current dataset from the recorded file
    if args.finetune_lambda is not None:   # When finetune_lambda is specified, directly use the specified lambda value for finetuning
        lam = args.finetune_lambda
    else:  # When finetune_lambda is not specified, read the corresponding lambda value from the file
        lam = get_best_lambda(finetune_file, args.dataset)
        print("Best lambda loaded:", lam)
    


        
    #====================Load Constraint Set for Training===================
    # Load constraints
    cons_path = f'./exp_{args.expName}/lab_VolMaxDCC/savedCons/{args.dataset}_{args.consRule}'
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



    #=====================Load Constraint Set for Testing====================
    if args.valCons_on_testset == "True":
        cons_path_test = f'./exp_{args.expName}/lab_VolMaxDCC/savedCons_test/{args.dataset}_balance' 
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
    # Train the VolMaxDCC model
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
        X = X,
        y = y, 
        test_X = test_X,
        test_y = test_y,
        lr = args.lr, 
        lam = lam,
        batch_size = args.batch_size, 
        num_epochs = args.epochs,
        tol = args.tol)
    
    #=======================When finetune_lambda is specified, record the ACC result on the validation set=======================
    if args.finetune_lambda is not None:   # When finetune_lambda is specified, record val_acc
        # Test the model on the validation set
        val_y = val_y.cpu().numpy()
        q_val = model.soft_assign(val_X)
        y_pred_val = model.predict(q_val)
        val_acc = model.evaluate(val_y, y_pred_val)
        # Record dataset, lambda, val_acc to the file
        update_finetune_file(finetune_file, args.dataset, args.finetune_lambda, val_acc)
    
    #======================When finetune_lambda is not specified, save the final results======================
    else:  # When finetune_lambda is not specified, normally record experiment results
        result_dir = os.path.join(lab_result_path, f"result_VolMaxDCC_{args.dataset}_{args.consRule}.csv")
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
