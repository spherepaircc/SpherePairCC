import sys
sys.path.append("..")
sys.path.append('./')
import argparse
import lib.loadData
from lib.loadData import *
from lib.consRules import *
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create constraints based on rules')
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")                  # 'extraCLs' or 'balance'
    parser.add_argument('--set', type=str, default="train")
    parser.add_argument('--orig_num', type=int, default=None)
    parser.add_argument('--extra_num', type=int, default=None)
    parser.add_argument('--J', type=int, default=10)
    parser.add_argument('--imbCluster', type=int, default=None)                     # Specify which cluster to use for generating extra constraints
    parser.add_argument('--modelVersion', type=str, default="CIDEC_Pretrain")       # Specify the model version, used to determine the constraint storage path
    parser.add_argument('--expName', type=str, default="anchor_issues")             # Specify the experiment name, used to determine the constraint storage path
    parser.add_argument('--finetune', type=str, default="False")                    # In finetune mode, set split_val=True when loading the dataset to ensure that the generated constraints do not include validation set samples
    args = parser.parse_args()


    #====================Load Dataset====================
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
            if args.finetune == "True":
                X, y, test_X, test_y, *_ = load_function(split_val=True)
            else:
                X, y, test_X, test_y, *_ = load_function()
        else:
            raise ValueError(f"Dataset loader for '{dataset_name}' not found.")


    #====================Generate Constraint Set====================
    if args.orig_num is None and args.extra_num is None:
        # If orig_num and extra_num are not specified, default to generating 0.1N~N constraints
        orig_num = len(y) * 0.1
        extra_num = len(y) * 0.9
    else:
        orig_num = args.orig_num
        extra_num = args.extra_num
    J = args.J
    imbCluster = args.imbCluster

    if args.set == "train":    # Training set constraints
        this_y = y
        consPath = f'./exp_{args.expName}/lab_{args.modelVersion}/savedCons/{args.dataset}_{args.consRule}'
    elif args.set == "test":   # Test set constraints
        this_y = test_y
        consPath = f'./exp_{args.expName}/lab_{args.modelVersion}/savedCons_test/{args.dataset}_{args.consRule}'
    else:
        raise ValueError(f"Set '{args.set}' not found, should be train or test.")

    # Generate constraint set based on the constraint generation rules
    if args.consRule == "extraCLs":
        generate_extraCLs(consPath, this_y, orig_num, extra_num, J, imbCluster)
    elif args.consRule == "balance":
        generate_balance(consPath, this_y, orig_num, extra_num, J)
    else:
        raise ValueError(f"Constraint rule '{args.consRule}' not found.")
