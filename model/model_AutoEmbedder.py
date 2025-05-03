import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import math
import csv
import os
from lib.utils import acc
import fastcluster
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score


def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if i < (len(layers) - 1):  
            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class AutoEmbedder(nn.Module):

    def __init__(self, 
                 input_dim=784,
                 z_dim=10,            # latent space dim
                 n_clusters=10,
                 encodeLayer=[500, 500, 2000],
                 decodeLayer=[2000, 500, 500],
                 activation="relu",
                 alpha=100.0):
        super(AutoEmbedder, self).__init__()

        # -----------------------------------------------------
        # define: encoder1, decoder1, enc_mu1, dec1
        # -----------------------------------------------------
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.alpha = alpha   # margin for CL
        
        # encoder
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        # decoder (though autoembedder does not use reconstruction loss, but keep it for compatibility)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        # -----------------------------------------------------
        #  define siamese: encoder2, decoder2, enc_mu2, dec2
        # -----------------------------------------------------
        self.encoder2 = buildNetwork([input_dim] + encodeLayer, activation=activation)
        self._enc_mu2 = nn.Linear(encodeLayer[-1], z_dim)

        self.decoder2 = buildNetwork([z_dim] + decodeLayer, activation=activation)
        self._dec2 = nn.Linear(decodeLayer[-1], input_dim)

        # placeholder for cluster centers
        self.W = Parameter(torch.Tensor(n_clusters, z_dim))
        nn.init.xavier_uniform_(self.W)

        if torch.cuda.is_available():
            self.cuda()



    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def forward_encoder1(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        return z

    def forward_encoder2(self, x):
        h = self.encoder2(x)
        z = self._enc_mu2(h)
        return z

    def forward_decoder1(self, z):
        h = self.decoder(z)
        x_recon = self._dec(h)
        return x_recon

    def forward_decoder2(self, z):
        h = self.decoder2(z)
        x_recon = self._dec2(h)
        return x_recon


    # Euclidean
    def pairwise_distance(self, z1, z2):
        dist = torch.sqrt(torch.sum((z1 - z2) ** 2, dim=1) + 1e-12)
        return dist

    # 2 margin ( margin for CL is alpha, margin for ML always be 10000) MSE loss
    def pairwise_loss(self, z1, z2, type_values):
        dist = self.pairwise_distance(z1, z2)
        ml_mask = (type_values == 1)
        cl_mask = (type_values == 0)
        
        # ML
        if ml_mask.sum() > 0:
            clipped = torch.clamp(dist[ml_mask], min=0, max=10000)   # use 10000 as margin for ML, otherwise gradient issue
            target_ml = 0.0
            loss_ml = torch.sum((clipped - target_ml) ** 2)
        else:
            loss_ml = 0.0
        
        # CL
        if cl_mask.sum() > 0:
            clipped = torch.clamp(dist[cl_mask], min=0, max=self.alpha)
            target_cl = self.alpha
            loss_cl = torch.sum((clipped - target_cl) ** 2)
        else:
            loss_cl = 0.0
        
        total_pairs = ml_mask.sum() + cl_mask.sum()
        if total_pairs == 0:
            return torch.tensor(0.0, device=dist.device)
        loss = (loss_ml + loss_cl) / total_pairs
        return loss

    

    # ---------------------- Constraint Data Processing ----------------------
    def merge_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
        """
        Mix and shuffle must-link (ML) and cannot-link (CL) constraints, and randomly swap the order of each pair of indices with a 50% probability.
        
        Parameters:
            ml_ind1 (numpy.ndarray): First index array for must-link constraints
            ml_ind2 (numpy.ndarray): Second index array for must-link constraints
            cl_ind1 (numpy.ndarray): First index array for cannot-link constraints
            cl_ind2 (numpy.ndarray): Second index array for cannot-link constraints
        
        Returns:
            numpy.ndarray: Mixed and randomly shuffled constraint array with shape (num_constraints, 3)
                        Each row is formatted as [index1, index2, label], where label is 1 for must-link and 0 for cannot-link
        """
        # Set must-link constraint labels to 1 and cannot-link constraint labels to 0
        ml_labels = np.ones(len(ml_ind1))
        cl_labels = np.zeros(len(cl_ind1))
        
        # Combine must-link and cannot-link constraints
        ml_cons = np.column_stack((ml_ind1, ml_ind2, ml_labels))
        cl_cons = np.column_stack((cl_ind1, cl_ind2, cl_labels))
        
        # Randomly swap the order of each constraint pair with a 50% probability
        def random_swap(cons):
            # cons: (num_pairs, 3)
            num_pairs = cons.shape[0]
            swap_mask = np.random.rand(num_pairs) < 0.5  # Generate random boolean mask
            # For rows to be swapped, exchange the first and second columns
            cons[swap_mask, 0], cons[swap_mask, 1] = cons[swap_mask, 1], cons[swap_mask, 0].copy()
            return cons
        
        ml_cons = random_swap(ml_cons)
        cl_cons = random_swap(cl_cons)
        
        # Merge all constraints
        mergedCons = np.concatenate((ml_cons, cl_cons), axis=0)
        
        # Shuffle the order of the merged constraints
        shuffled_indices = np.random.permutation(mergedCons.shape[0])
        shuffledMergedCons = mergedCons[shuffled_indices]
        
        return shuffledMergedCons

    def satisfied_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred):
        """
        Same as in SpherePairs, used to record the proportion of constraints satisfied during clustering.
        """
        if ml_ind1.size == 0 or ml_ind2.size == 0 or cl_ind1.size == 0 or cl_ind2.size == 0:
            return 1.1
        count = 0
        satisfied = 0
        for (i, j) in zip(ml_ind1, ml_ind2):
            count += 1
            if y_pred[i] == y_pred[j]:
                satisfied += 1
        for (i, j) in zip(cl_ind1, cl_ind2):
            count += 1
            if y_pred[i] != y_pred[j]:
                satisfied += 1
        return float(satisfied)/count


    # ---------------------- Obtain Embedding (Using Only encoder1) ----------------------
    def encodeBatch(self, X, batch_size=256):
        """
        Similar to encodeBatch in SpherePairs.
        Uses only the first branch (encoder1) to obtain embeddings of samples.
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            if use_cuda:
                inputs = inputs.cuda()
            z = self.forward_encoder1(inputs)
            encoded.append(z.data)
        encoded = torch.cat(encoded, dim=0)
        return encoded.cpu()



    # ---------------------- Training: fit (Retain parameter order compatible with SpherePairs) ----------------------
    def fit(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2,
        X, 
        y=None,
        lr=1e-3, 
        batch_size=256, 
        epochs=500,
        soft_epochs=100,
        tol=1e-3,
        record_feature_dir=None,
        record_feature_norm=False
    ):
        print("===================== Train AutoEmbedder on pairwise constraints =====================")
        
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if y is not None:
            y = y.cpu().numpy()

        # Merge constraints
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long()
        if use_cuda:
            mergedCons = mergedCons.cuda()

        mergedCons_num = mergedCons.shape[0]
        mergedCons_num_batch = int(math.ceil(1.0 * mergedCons_num / batch_size))

        # Optimizer only optimizes encoder1/encoder2
        optimizer_params = optim.Adam(
            list(self.encoder.parameters()) + list(self._enc_mu.parameters()) +
            list(self.encoder2.parameters()) + list(self._enc_mu2.parameters()),
            lr=lr, weight_decay=0
        )

        min_loss = 1e10
        last_loss = 1e10
        count_stop = 0
        final_epoch = 0

        for epoch in range(epochs):
            final_epoch = epoch

            # ------------ Evaluate training loss (no update) ------------
            self.eval()
            with torch.no_grad():
                px1 = X[mergedCons[:, 0]]
                px2 = X[mergedCons[:, 1]]
                type_values = mergedCons[:, 2]
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                if use_cuda:
                    inputs1 = inputs1.cuda()
                    inputs2 = inputs2.cuda()
                    type_values = type_values.cuda()
                z1 = self.forward_encoder1(inputs1)
                z2 = self.forward_encoder2(inputs2)
                loss_pairwise_train = self.pairwise_loss(z1, z2, type_values)
                delta = (loss_pairwise_train.item() - min_loss) / (min_loss + 1e-12)
                if loss_pairwise_train.item() < min_loss:
                    min_loss = loss_pairwise_train.item()
                if epoch > soft_epochs and abs(delta) < tol:
                    break
                # # observed no difference
                # if abs(last_loss - loss_pairwise_train.item())/last_loss < 0.1:
                #     count_stop += 1
                # else:
                #     count_stop = 0
                # if epoch > soft_epochs and count_stop >= 5:
                #     break
            
            # ------------ Evaluate feature norm (no update) ------------
            if record_feature_norm:
                with torch.no_grad():
                    inputs = Variable(X)
                    if use_cuda:
                        inputs = inputs.cuda()
                    z = self.forward_encoder1(inputs)
                    norm_avg = torch.mean(torch.norm(z, p=2, dim=1))
                    norm_std = torch.std(torch.norm(z, p=2, dim=1))


            # ------------ Training ------------
            self.train()
            shuffle_idx = np.random.permutation(mergedCons_num)
            mergedCons = mergedCons[shuffle_idx]
            total_pairwise_loss = 0.0

            for bidx in range(mergedCons_num_batch):
                bstart = bidx * batch_size
                bend = min(mergedCons_num, (bidx + 1) * batch_size)
                px1 = X[mergedCons[bstart: bend, 0]]
                px2 = X[mergedCons[bstart: bend, 1]]
                type_values = mergedCons[bstart: bend, 2]

                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                if use_cuda:
                    inputs1 = inputs1.cuda()
                    inputs2 = inputs2.cuda()
                    type_values = type_values.cuda()

                optimizer_params.zero_grad()
                z1 = self.forward_encoder1(inputs1)
                z2 = self.forward_encoder2(inputs2)
                loss_pairwise = self.pairwise_loss(z1, z2, type_values)
                loss_pairwise.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer_params.step()

                total_pairwise_loss += loss_pairwise.item() * len(inputs1)

            avg_pair_loss = total_pairwise_loss / mergedCons_num
            print(f"(Embedding) Epoch {epoch}, Pairwise Loss: {avg_pair_loss:.5f}")

            # ------------ Log Recording ------------
            record_list = [
                epoch,               # epoch
                avg_pair_loss,       # loss_pairwise_train
                0, 0,                # loss_recon_train=0, loss_pair_recon_train=0 (placeholders to align with SpherePair)
                0, 0,                # acc_train, acc_test
                0, 0,                # nmi_train, nmi_test
                0, 0                 # ari_train, ari_test
            ]
            if record_feature_norm:
                record_list.extend([norm_avg.item(), norm_std.item()])

            if record_feature_norm:
                if not os.path.exists(record_log_dir): 
                    with open(record_log_dir, "w") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["epoch", 
                                        "loss_pairwise_train", "loss_recon_train", "loss_pair_recon_train",
                                        "acc_train", "acc_test", 
                                        "nmi_train", "nmi_test",
                                        "ari_train", "ari_test",
                                        "norm_avg", "norm_std"])
            else:
                if not os.path.exists(record_log_dir):
                    with open(record_log_dir, "w") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["epoch", 
                                        "loss_pairwise_train", "loss_recon_train", "loss_pair_recon_train",
                                        "acc_train", "acc_test", 
                                        "nmi_train", "nmi_test",
                                        "ari_train", "ari_test"])
            with open(record_log_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(record_list)

        # ------------ Save Features if Needed ------------
        if record_feature_dir is not None:
            latent = self.encodeBatch(X)
            torch.save(latent, record_feature_dir)
            print(f"Saved embeddings to {record_feature_dir}.")

        return final_epoch


    # ---------------------- Unsupervised Clustering: KMeans ----------------------
    def assign_Kmeans(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2,
        X, y=None,
        test_X=None, test_y=None,
        specific_K=None,
        n_init=50
    ):
        """
        Parameter signature aligned with SpherePairs.assign_Kmeans().
        Performs KMeans only in Euclidean space and assigns the centers to self.W.
        """
        print("====================== Use Kmeans to find W for assignments ======================")
        list_log = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        silhouette_score_avg = None 

        # Get training set embeddings
        data = self.encodeBatch(X)
        data_np = data.numpy()
        
        if specific_K is not None:
            # ============================ Manually set specific K to determine W, calculate Silhouette Score for experimental records ============================ #
            self.n_clusters = specific_K
            self.W = Parameter(torch.Tensor(self.n_clusters, self.z_dim))
            nn.init.xavier_uniform_(self.W)

            sample_size = 5000
            best_kmeans = None
            best_inertia = float('inf')
            silhouette_scores = []
            sc_n_init = 5  # Number of runs for each specific K to calculate the average Silhouette Score
            for i in range(sc_n_init):
                kmeans = KMeans(self.n_clusters, n_init=1)  # Run only once each time
                y_pred = kmeans.fit_predict(data_np)
                score = silhouette_score(data_np, y_pred, sample_size=sample_size)  # Randomly sample to calculate Silhouette Score
                silhouette_scores.append(score)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
                print(f"Kmeans run {i+1}/{sc_n_init} for specific_K={specific_K}, inertia: {kmeans.inertia_}, silhouette score: {score:.5f}")
            average_score = np.mean(silhouette_scores)   # Average Silhouette Score over multiple runs
            silhouette_score_avg = average_score   # Average Silhouette Score
            silhouette_score_std = np.std(silhouette_scores)   # Standard deviation of Silhouette Scores
            print(f"Average Silhouette Score for specific_K={specific_K}: {silhouette_score_avg:.5f}")   # Average Silhouette Score
            with torch.no_grad():
                self.W.data.copy_(torch.Tensor(best_kmeans.cluster_centers_))  # Directly update centroid W
        else:
            # ============================= Automatically determine W using ground truth K, no need to record Silhouette Score ============================ #
            kmeans = KMeans(self.n_clusters, n_init=n_init)
            y_pred = kmeans.fit_predict(data_np)
            # Update clustering centers to self.W
            with torch.no_grad():
                self.W.data.copy_(torch.from_numpy(kmeans.cluster_centers_))

        # Constraint satisfaction
        ml_ind1 = ml_ind1.astype(np.int64)
        ml_ind2 = ml_ind2.astype(np.int64)
        cl_ind1 = cl_ind1.astype(np.int64)
        cl_ind2 = cl_ind2.astype(np.int64)
        satisfied_cons = self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred)
        print(f"satisfied constraints: {satisfied_cons:.5f}")

        # Training set performance
        final_acc, final_nmi, final_ari = 0, 0, 0
        if y is not None:
            y = y.cpu().numpy()
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)
            print(f"Train set => ACC: {final_acc:.5f}, NMI: {final_nmi:.5f}, ARI: {final_ari:.5f}")

        # Test set performance
        test_acc, test_nmi, test_ari = 0, 0, 0
        if test_X is not None and test_y is not None:
            test_data = self.encodeBatch(test_X).numpy()
            centers = kmeans.cluster_centers_
            dist = np.sum((test_data[:, None, :] - centers[None, :, :])**2, axis=2)
            test_y_pred = np.argmin(dist, axis=1)

            test_y = test_y.cpu().numpy()
            test_acc = acc(test_y, test_y_pred)
            test_nmi = normalized_mutual_info_score(test_y, test_y_pred)
            test_ari = adjusted_rand_score(test_y, test_y_pred)
            print(f"Test set => ACC: {test_acc:.5f}, NMI: {test_nmi:.5f}, ARI: {test_ari:.5f}")

        # Log
        list_log[4] = final_acc
        list_log[5] = test_acc
        list_log[6] = final_nmi
        list_log[7] = test_nmi
        list_log[8] = final_ari
        list_log[9] = test_ari

        if not os.path.exists(record_log_dir):
            with open(record_log_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch",
                                "loss_pairwise_train", "loss_recon_train", "loss_pair_recon_train",
                                "acc_train", "acc_test", 
                                "nmi_train", "nmi_test",
                                "ari_train", "ari_test"])
        with open(record_log_dir, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list_log)

        if specific_K is not None:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari, silhouette_score_avg, silhouette_score_std
        else:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari


    # ---------------------- Unsupervised Clustering: Hierarchical ----------------------
    def assign_Hierarchical(self,
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2,
        X, y=None,
        test_X=None, test_y=None,
        specific_K=None,
        method='ward'
    ):
        """
        Parameter signature aligned with SpherePairs.assign_Hierarchical().
        Uses hierarchical clustering in Euclidean space to perform fcluster on embeddings, then updates self.W based on cluster centers.
        """
        print("====================== Use Hierarchical Clustering to find W for assignments ======================")
        list_log = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        silhouette_score_value = None 

        if specific_K is not None:
            self.n_clusters = specific_K
            self.W = Parameter(torch.Tensor(self.n_clusters, self.z_dim))
            nn.init.xavier_uniform_(self.W)

        data = self.encodeBatch(X)
        data_np = data.numpy()

        Z = fastcluster.linkage(data_np, method=method)
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')  # 1 ~ n_clusters

        # Calculate cluster centers
        centers = np.zeros((self.n_clusters, data_np.shape[1]), dtype=np.float32)
        for c in range(self.n_clusters):
            cluster_points = data_np[labels == (c + 1)]
            if len(cluster_points) > 0:
                centers[c] = cluster_points.mean(axis=0)
            else:
                centers[c] = 0.0
        with torch.no_grad():
            self.W.data.copy_(torch.from_numpy(centers))

        # If specific_K is specified, calculate Silhouette Score as a simple reference
        if specific_K is not None:
            sample_size = min(5000, len(data_np))
            silhouette_score_value = silhouette_score(data_np, labels, sample_size=sample_size, random_state=42)
            print(f"Silhouette Score for Hierarchical Clustering (K={specific_K}): {silhouette_score_value:.5f}")
        
        ml_ind1 = ml_ind1.astype(np.int64)
        ml_ind2 = ml_ind2.astype(np.int64)
        cl_ind1 = cl_ind1.astype(np.int64)
        cl_ind2 = cl_ind2.astype(np.int64)
        y_pred = labels - 1  # Convert to 0~(K-1)
        satisfied_cons = self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred)
        print(f"satisfied constraints: {satisfied_cons:.5f}")

        # Training set performance
        final_acc, final_nmi, final_ari = 0, 0, 0
        if y is not None:
            y = y.cpu().numpy()
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)
            print(f"Train set => ACC: {final_acc:.5f}, NMI: {final_nmi:.5f}, ARI: {final_ari:.5f}")

        # Test set performance
        test_acc, test_nmi, test_ari = 0, 0, 0
        if test_X is not None and test_y is not None:
            test_data = self.encodeBatch(test_X)
            test_data_np = test_data.numpy()

            dist = np.sum((test_data_np[:, None, :] - centers[None, :, :])**2, axis=2)
            test_y_pred = np.argmin(dist, axis=1)
            test_y = test_y.cpu().numpy()

            test_acc = acc(test_y, test_y_pred)
            test_nmi = normalized_mutual_info_score(test_y, test_y_pred)
            test_ari = adjusted_rand_score(test_y, test_y_pred)
            print(f"Test set => ACC: {test_acc:.5f}, NMI: {test_nmi:.5f}, ARI: {test_ari:.5f}")

        # Log
        list_log[4] = final_acc
        list_log[5] = test_acc
        list_log[6] = final_nmi
        list_log[7] = test_nmi
        list_log[8] = final_ari
        list_log[9] = test_ari

        if not os.path.exists(record_log_dir):
            with open(record_log_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch",
                                "loss_pairwise_train", "loss_recon_train", "loss_pair_recon_train",
                                "acc_train", "acc_test", 
                                "nmi_train", "nmi_test",
                                "ari_train", "ari_test"])
        with open(record_log_dir, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list_log)

        if specific_K is not None:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari, silhouette_score_value
        else:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari
