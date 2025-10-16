import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS as MDS_skl
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.stats import ks_2samp



def fit_dimensionality_reduction(z_fit_data, z_cl_np, max_components, method='PCA'):    

    if method == 'PCA':
        print("Step 2.5: Using standard PCA (with centering)...")
        pca = PCA(n_components=max_components)
        pca.fit(z_fit_data)
        z_cl_reduced_full = pca.transform(z_cl_np)
        max_valid_components = max_components
        
    elif method == 'TruncatedSVD':
        print("Step 2.5: Using TruncatedSVD (no centering)...")
        if max_components >= z_fit_data.shape[1]:
            max_valid_components = z_fit_data.shape[1] - 1
        else:
            max_valid_components = max_components
        pca = TruncatedSVD(n_components=max_valid_components)
        pca.fit(z_fit_data)
        z_cl_reduced_full = pca.transform(z_cl_np)
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return z_cl_reduced_full, max_valid_components


def calculate_angle_metrics_simplified(z_cl_reduced_norm, cl_ind1, cl_ind2, idx_map, ratio=0.05):
    angles = []
    for i in range(len(cl_ind1)):
        idx1 = cl_ind1[i]
        idx2 = cl_ind2[i]
        vec1 = z_cl_reduced_norm[idx_map[idx1]]
        vec2 = z_cl_reduced_norm[idx_map[idx2]]
        cos_sim = np.dot(vec1, vec2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle = np.arccos(cos_sim)
        angles.append(angle)
    angles = np.array(angles)

    if len(angles) > 0:
        angle_mean = np.mean(angles)
        sorted_angles = np.sort(angles)
        n_ratio = max(1, int(ratio * len(sorted_angles)))
        min_ratio_mean = np.mean(sorted_angles[:n_ratio])
        diff = angle_mean - min_ratio_mean
        return angle_mean, min_ratio_mean, diff
    else:
        return np.nan, np.nan, np.nan


def analyze_K_with_pca_variance_core(z_np, cl_ind1, cl_ind2, max_components=None, ratio=0.05, sample_size=None, method='PCA'):

    feature_dim = z_np.shape[1]

    if max_components is None:
        max_components = feature_dim
    elif max_components > feature_dim:
        print(f"Warning: max_components ({max_components}) > feature_dim ({feature_dim}). Clamping to {feature_dim}.")
        max_components = feature_dim

    cl_indices = np.unique(np.concatenate((cl_ind1, cl_ind2)))
    z_cl_np = z_np[cl_indices]
    if sample_size is not None:
        num_total_samples = z_np.shape[0]
        sample_size = min(sample_size, num_total_samples)
        print(f"Step 2: Sampling {sample_size} points from the dataset to fit model...")
        sample_indices = np.random.choice(num_total_samples, sample_size, replace=False)
        z_fit_data = z_np[sample_indices]
    else:
        print(f"Step 2: Using all {z_cl_np.shape[0]} unique constraint-involved samples to fit model...")
        z_fit_data = z_cl_np

    z_cl_reduced_full, max_valid_components = fit_dimensionality_reduction(
        z_fit_data, z_cl_np, max_components, method
    )

    idx_map = {original_idx: new_idx for new_idx, original_idx in enumerate(cl_indices)}

    angle_means = []
    min_ratio_means = []
    diffs = []
    all_angles_list = [] 
    
    if max_valid_components == feature_dim:
        z_cl_norm_original = normalize(z_cl_reduced_full, norm='l2', axis=1)
    else:
        z_cl_norm_original = normalize(z_cl_np, norm='l2', axis=1)
    
    original_angles = []
    for i in range(len(cl_ind1)):
        idx1 = cl_ind1[i]
        idx2 = cl_ind2[i]
        vec1 = z_cl_norm_original[idx_map[idx1]]
        vec2 = z_cl_norm_original[idx_map[idx2]]
        cos_sim = np.dot(vec1, vec2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle = np.arccos(cos_sim)
        original_angles.append(angle)
    original_angles = np.array(original_angles)
    
    print(f"Step 3: Calculating metrics for dimensions 1 to {max_valid_components}...")

    for d in range(1, max_valid_components + 1):
        if d == max_valid_components and max_valid_components == feature_dim:
            z_cl_reduced_norm = z_cl_norm_original
        else:
            z_cl_reduced = z_cl_reduced_full[:, :d]
            z_cl_reduced_norm = normalize(z_cl_reduced, norm='l2', axis=1)

        angle_mean, min_ratio_mean, diff_old = calculate_angle_metrics_simplified(
            z_cl_reduced_norm, cl_ind1, cl_ind2, idx_map, ratio
        )

        if not np.isnan(angle_mean):
            angle_means.append(f"{angle_mean / np.pi:.4f}pi")
            min_ratio_means.append(f"{min_ratio_mean / np.pi:.4f}pi")
        else:
            angle_means.append("nan")
            min_ratio_means.append("nan")

        if d == max_valid_components and max_valid_components == feature_dim:
            current_angles = original_angles
        else:
            current_angles = []
            for i in range(len(cl_ind1)):
                idx1 = cl_ind1[i]
                idx2 = cl_ind2[i]
                vec1 = z_cl_reduced_norm[idx_map[idx1]]
                vec2 = z_cl_reduced_norm[idx_map[idx2]]
                cos_sim = np.dot(vec1, vec2)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                angle = np.arccos(cos_sim)
                current_angles.append(angle)
            current_angles = np.array(current_angles)
        all_angles_list.append(current_angles)


        curr_angles = all_angles_list[d-1] 
        
        if len(curr_angles) > 0:
            sorted_angles = np.sort(curr_angles)
            n_ratio = max(1, int(ratio * len(sorted_angles)))
            min_angles = sorted_angles[:n_ratio]
            
            min_angles_mean = np.mean(min_angles)
            diffs.append(f"{min_angles_mean / np.pi:.4f}pi")
        else:
            diffs.append("nan")


        if (d % 10 == 0) or (d == max_valid_components):
            print(f"  - Dim {d}: AngleMean={angle_means[-1]}, MinRatioMean={min_ratio_means[-1]}, Diff={diffs[-1]}")

    # If TruncatedSVD leads to incomplete dimensions, append the last one
    if method == 'TruncatedSVD' and max_valid_components < feature_dim:
        print("Appending stats for the original space (un-reduced)...")
        z_cl_norm = normalize(z_cl_np, norm='l2', axis=1)
        
        angle_mean, min_ratio_mean, diff_old = calculate_angle_metrics_simplified(
            z_cl_norm, cl_ind1, cl_ind2, idx_map, ratio
        )

        if not np.isnan(angle_mean):
            angle_means.append(f"{angle_mean / np.pi:.4f}pi")
            min_ratio_means.append(f"{min_ratio_mean / np.pi:.4f}pi")
        else:
            angle_means.append("nan")
            min_ratio_means.append("nan")

        original_angles_truncated = original_angles

        if not np.isnan(angle_mean):
            if len(original_angles) > 0:
                sorted_angles = np.sort(original_angles)
                n_ratio = max(1, int(ratio * len(sorted_angles)))
                min_angles = sorted_angles[:n_ratio]
                
                min_angles_mean = np.mean(min_angles)
                diffs.append(f"{min_angles_mean / np.pi:.4f}pi")
            else:
                diffs.append("nan")
        else:
            diffs.append("nan")

        print(f"  - Dim {feature_dim}: AngleMean={angle_means[-1]}, MinRatioMean={min_ratio_means[-1]}, Diff={diffs[-1]}")

    return angle_means, min_ratio_means, diffs