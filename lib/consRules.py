import numpy as np
import os
import torch



#=================================================================================================
# Generate J nested incremental constraint sets / extraCLs
def generate_extraCLs(consPath, y, orig_num, extra_num, J, imbCluster):
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    labels = np.unique(y)
    n_samples = len(y)
    sample_indices = np.arange(n_samples)
    all_constraints = []

    # Generate orig_num original constraints and store them in all_constraints
    seen_pairs = set()
    orig_ml, orig_cl = [], []
    while len(orig_ml) + len(orig_cl) < orig_num:
        pair = tuple(np.random.choice(sample_indices, 2, replace=False)) 
        if pair not in seen_pairs and pair[0] != pair[1]:
            seen_pairs.add(pair)
            if y[pair[0]] == y[pair[1]]:
                orig_ml.append(pair)
            else:
                orig_cl.append(pair)
    all_constraints.append((orig_ml, orig_cl))

    # Generate extra_num additional constraints and store them in all_constraints
    if imbCluster is not None:
        chosen_class = imbCluster  # Use the category specified by imbCluster to generate extra constraints
        chosen_class_indices = sample_indices[y == chosen_class]
    else:
        chosen_class = np.random.choice(labels)  # Randomly select a category
        chosen_class_indices = sample_indices[y == chosen_class]
    other_class_indices = sample_indices[y != chosen_class]

    extra_pairs = []
    for j in range(2, J+1):
        extra_num_j = (extra_num // (J - 1)) * (j - 1)
        while len(extra_pairs) < extra_num_j:
            sample_1 = np.random.choice(chosen_class_indices)          # Select sample_1 from the specified category
            sample_2 = np.random.choice(other_class_indices)           # Select sample_2 from the remaining categories
            pair = (sample_1, sample_2) if sample_1 < sample_2 else (sample_2, sample_1)
            if pair not in seen_pairs and pair[0] != pair[1]:
                seen_pairs.add(pair)
                extra_pairs.append(pair)
        # Temporarily save this constraint set (contains "orig constraints" and "extra constraints so far")
        this_ml = orig_ml.copy()
        this_cl = orig_cl.copy()
        for p in extra_pairs:
            if y[p[0]] == y[p[1]]:
                this_ml.append(p)
            else:
                this_cl.append(p)
        all_constraints.append((this_ml, this_cl))

    # Save the constraint sets to files
    os.makedirs(consPath, exist_ok=True)
    for idx, constraints in enumerate(all_constraints, start=1):
        np.savez(f'{consPath}/constraints_{idx}.npz', ml=np.array(constraints[0]), cl=np.array(constraints[1]))
#=================================================================================================



#=================================================================================================
# Generate J nested incremental constraint sets / balance
def generate_balance(consPath, y, orig_num, extra_num, J):
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    n_samples = len(y)
    sample_indices = np.arange(n_samples)
    all_constraints = []

    # Generate orig_num original constraints and store them in all_constraints
    seen_pairs = set()
    orig_ml, orig_cl = [], []
    while len(orig_ml) + len(orig_cl) < orig_num:
        pair = tuple(np.random.choice(sample_indices, 2, replace=False))
        if pair not in seen_pairs and pair[0] != pair[1]:
            seen_pairs.add(pair)
            if y[pair[0]] == y[pair[1]]:
                orig_ml.append(pair)
            else:
                orig_cl.append(pair)
    all_constraints.append((orig_ml, orig_cl))

    # Generate extra_num additional constraints and store them in all_constraints
    extra_pairs = []
    for j in range(2, J+1):
        extra_num_j = (extra_num // (J - 1)) * (j - 1)
        while len(extra_pairs) < extra_num_j:
            sample_1 = np.random.choice(sample_indices)     # Select from all samples
            sample_2 = np.random.choice(sample_indices)     # Select from all samples
            pair = (sample_1, sample_2) if sample_1 < sample_2 else (sample_2, sample_1)
            if pair not in seen_pairs and pair[0] != pair[1]:
                seen_pairs.add(pair)
                extra_pairs.append(pair)
        # Temporarily save this constraint set (contains "orig constraints" and "extra constraints so far")
        this_ml = orig_ml.copy()
        this_cl = orig_cl.copy()
        for p in extra_pairs:
            if y[p[0]] == y[p[1]]:
                this_ml.append(p)
            else:
                this_cl.append(p)
        all_constraints.append((this_ml, this_cl))

    # Save the constraint sets to files
    os.makedirs(consPath, exist_ok=True)
    for idx, constraints in enumerate(all_constraints, start=1):
        np.savez(f'{consPath}/constraints_{idx}.npz', ml=np.array(constraints[0]), cl=np.array(constraints[1]))
#=================================================================================================

