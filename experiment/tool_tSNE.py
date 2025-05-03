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
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='baseline: VanillaDCC')
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--consRule', type=str, default="balance")
    parser.add_argument('--consIndex', type=int, default=1)
    parser.add_argument('--modelVersion', type=str, default="CIDEC_Pretrain")     # Specify the model version, used to determine the constraint storage path
    args = parser.parse_args()

    #======================Load y from Dataset=======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_name = args.dataset
    sub_pattern = re.compile(r'^(?P<base>\w+)_sub_(?P<n_class>\d+)$')   # Regular expression to match 'dataset_sub_n'
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


    #======================Load Saved Features=======================
    lab_result_path = f"./exp_tSNE/lab_{args.modelVersion}/{args.dataset}/{args.consRule}"
    prefix = f"feature_{args.dataset}_{args.consRule}_{args.consIndex}"
    pattern = os.path.join(lab_result_path, prefix + "*.pt")
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"no file match: {pattern}")
    max_num = -1
    latest_file = None
    for file_path in matching_files:
        base_name = os.path.basename(file_path)
        parts = base_name.split('_')
        if len(parts) >= 5 and parts[-1].endswith(".pt"):
            num_part = parts[-1][:-3]
            try:
                num = int(num_part)
                if num > max_num:
                    max_num = num
                    latest_file = file_path
            except ValueError:
                continue
    if latest_file is None:
        raise FileNotFoundError(f"no file index match: {matching_files}")
    feature = torch.load(latest_file)
    feature = feature.to(device)


    #====================Load Constraint Set===================
    cons_path = f'./exp_tSNE/lab_{args.modelVersion}/savedCons/{args.dataset}_{args.consRule}'
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


    #====================Random Sampling of Constraint Set====================
    # Calculate the number of constraints used for plotting
    sample_rate = 0.005
    sample_size_ml = int(len(ml_ind1) * sample_rate)
    sample_size_cl = int(len(cl_ind1) * sample_rate)

    print(f"Total ML constraints: {len(ml_ind1)}, Sampled ML constraints: {sample_size_ml}")
    print(f"Total CL constraints: {len(cl_ind1)}, Sampled CL constraints: {sample_size_cl}")

    # Randomly sample from ML constraints
    if sample_size_ml > 0:
        ml_sample_indices = np.random.choice(len(ml_ind1), size=sample_size_ml, replace=False)
        sampled_ml_ind1 = ml_ind1[ml_sample_indices]
        sampled_ml_ind2 = ml_ind2[ml_sample_indices]
    else:
        sampled_ml_ind1 = np.array([])
        sampled_ml_ind2 = np.array([])

    # Randomly sample from CL constraints
    if sample_size_cl > 0:
        cl_sample_indices = np.random.choice(len(cl_ind1), size=sample_size_cl, replace=False)
        sampled_cl_ind1 = cl_ind1[cl_sample_indices]
        sampled_cl_ind2 = cl_ind2[cl_sample_indices]
    else:
        sampled_cl_ind1 = np.array([])
        sampled_cl_ind2 = np.array([])


    #====================Select a subset of samples for t-SNE dimensionality reduction====================
    np.random.seed(42) 
    total_samples = feature.size(0)
    subset_size = int(total_samples * 0.1)
    subset_size = 5000
    subset_indices = np.random.choice(total_samples, size=subset_size, replace=False)
    constraint_indices = np.unique(np.concatenate([sampled_ml_ind1, sampled_ml_ind2, sampled_cl_ind1, sampled_cl_ind2]))
    combined_indices = np.unique(np.concatenate([subset_indices, constraint_indices]))
    
    subset_features = feature[combined_indices].cpu().numpy()
    subset_labels = y[combined_indices].cpu().numpy()


    #====================Perform dimensionality reduction using t-SNE====================
    if args.modelVersion == "VanillaDCC":
        print("Performing PCA...")
        pca = PCA(n_components=10, random_state=42)  # First perform PCA to reduce to 10 dimensions
        pca_features = pca.fit_transform(subset_features)
        subset_features = pca_features
        # Use t-SNE to reduce features to 2 dimensions
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        subset_tsne = tsne.fit_transform(subset_features)
    else:
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        subset_tsne = tsne.fit_transform(subset_features)
        print("t-SNE completed.")
    
    index_to_tsne = {idx: coord for idx, coord in zip(combined_indices, subset_tsne)}
    
    #====================Prepare Plotting Data====================
    is_subset = np.isin(combined_indices, subset_indices)
    labels_subset = subset_labels
    
    #====================Plot t-SNE Visualization====================
    plt.figure(figsize=(12, 12))
    
    # Plot subset sample points, categorized by label
    unique_labels = np.unique(labels_subset[is_subset])
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for label in unique_labels:
        indices = (labels_subset == label) & is_subset
        plt.scatter(subset_tsne[indices, 0], subset_tsne[indices, 1], 
                    c=np.array([colors(label)]), 
                    # edgecolor='k', linewidth=0.3,
                    label=str(label), alpha=1, s=50)    # alpha=0.8, s=10
        
    
    # Print some mapping information for debugging
    if len(sampled_ml_ind1) > 0 or len(sampled_cl_ind1) > 0:
        print("Drawing constraints...")
    
    # Draw ML constraint lines (blue)
    for ind1, ind2 in zip(sampled_ml_ind1, sampled_ml_ind2):
        coord1 = index_to_tsne.get(ind1)
        coord2 = index_to_tsne.get(ind2)
        if coord1 is not None and coord2 is not None:
            plt.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'b-', linewidth=0.5, alpha=0.3)
    
    # Draw CL constraint lines (red)
    for ind1, ind2 in zip(sampled_cl_ind1, sampled_cl_ind2):
        coord1 = index_to_tsne.get(ind1)
        coord2 = index_to_tsne.get(ind2)
        if coord1 is not None and coord2 is not None:
            plt.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'r-', linewidth=0.5, alpha=0.3)
    
    # Thicken the image borders
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    
    output_dir = f'./exp_tSNE/lab_{args.modelVersion}/{args.dataset}/{args.consRule}'
    os.makedirs(output_dir, exist_ok=True)

    prefix = f'tSNE_visualization_{args.consIndex}'
    pattern = os.path.join(output_dir, prefix + "*.png")
    matching_files = glob.glob(pattern)

    if not matching_files:
        new_suffix = "_00"
    else:
        max_num = -1
        for file_path in matching_files:
            base_name = os.path.basename(file_path)
            parts = base_name.split('_')
            if parts[-1].endswith(".png"):
                num_part = parts[-1][:-4]
                try:
                    num = int(num_part)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
        new_suffix = f"_{max_num + 1:02d}"

    new_file_name_png = prefix + new_suffix + ".png"
    output_path_png = os.path.join(output_dir, new_file_name_png)

    # Save in png and pdf formats
    plt.savefig(output_path_png, dpi=300)
    plt.savefig(output_path_png.replace('.png', '.pdf'), dpi=300)
    plt.close()
