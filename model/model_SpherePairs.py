import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import math
import csv
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import silhouette_score
import fastcluster
from scipy.cluster.hierarchy import fcluster


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class SpherePairs(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self.n_clusters = n_clusters
        self.W = Parameter(torch.Tensor(n_clusters, z_dim))   # just placeholder for cluster centers


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)   # feature vector
        z_normalized = F.normalize(z, p=2, dim=1, eps=1e-12)
        h = self.decoder(z_normalized)
        xrecon = self._dec(h)   # decoded vector
        W = self.W  # using W for hard assignment (cluster centers)
        _, y_pred = self.hard_assign(z, W)   # hard assignment
        return z, xrecon, y_pred


    def hard_assign(self, z, W):
        cosine = F.linear(F.normalize(z), F.normalize(W)) 
        y_pred = torch.argmax(cosine, dim=1) 
        return cosine, y_pred


    # our angular pairwise loss
    def pairwise_loss(self, z1, z2, type_values, omega=2):
        sim = (F.cosine_similarity(z1, z2, eps=1e-8) + 1) / 2 
        cos_theta = 2 * sim - 1
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        phase_cos = torch.cos(omega * theta)
        phase_cos = phase_cos.clamp(-1, 1)
        phase_sim = torch.where(theta <= np.pi / omega, (phase_cos + 1) / 2, torch.zeros_like(phase_cos))
        adjusted_sim = torch.where(type_values == 1, sim, phase_sim)
        adjusted_sim = adjusted_sim.clamp(0, 1)
        if torch.isnan(adjusted_sim).any():
            print("NaNs detected in adjusted_sim")
        loss = F.binary_cross_entropy(adjusted_sim, type_values.float(), reduction='mean')
        return loss


    # reconstruction loss
    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon-x)**2)
        return recon_loss

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _ = self.forward(inputs)
            # normalize z
            z = F.normalize(z, p=2, dim=1)
            encoded.append(z.data)
        encoded = torch.cat(encoded, dim=0)
        return encoded


    def satisfied_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred):
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


    def predict(self, X, y):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        latent = self.encodeBatch(X)
        W = self.W
        _, y_pred = self.hard_assign(latent, W)
        y_pred = y_pred.data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)
        return final_acc, final_nmi, final_ari


    def merge_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
        ml_labels = np.ones(len(ml_ind1))
        cl_labels = np.zeros(len(cl_ind1))
        ml_cons = np.column_stack((ml_ind1, ml_ind2, ml_labels))
        cl_cons = np.column_stack((cl_ind1, cl_ind2, cl_labels))
        mergedCons = np.concatenate((ml_cons, cl_cons), axis=0)
        indices = np.random.permutation(np.arange(mergedCons.shape[0]))
        shuffledMergedCons = mergedCons[indices]
        return shuffledMergedCons


    # draw 3D sphere visualization of embedding vectors, only available when z_dim=3
    def visualize_3d_sphere(self, X, y, epoch=None, train_ACC=None, num_samples=2000, num_cons=200, save_path='3d_sphere.png', stage = "embedding", mergedCons=None):
        self.eval()
        indices = np.random.choice(X.size(0), num_samples, replace=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subset_X = X[indices].float().to(device)
        subset_y = torch.tensor(y[indices], dtype=torch.long).to(device)
        subset_y = subset_y.cpu().numpy()
        n_classes = np.max(y) + 1
        # get the embedding vectors z from the model
        with torch.no_grad():
            h = self.encoder(subset_X)
            z = self._enc_mu(h)
            z_normalized = F.normalize(z, p=2, dim=1).cpu().numpy()
        # only draw z in embedding stage, draw z and W in assign stage
        if stage == "assign":
            W = self.W
            W_normalized = F.normalize(W.clone().detach().cpu(), p=2, dim=1).numpy()        
        cmap = plt.get_cmap('tab10', n_classes)
        norm = BoundaryNorm(boundaries=np.arange(-0.5, n_classes + 0.5, 1), ncolors=n_classes)
        # 3D visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1, rstride=5, cstride=5)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3, linewidth=0.5)
        z_values = z_normalized[:, 2]
        front_mask = z_values >= 0
        back_mask = z_values < 0
        scatter_front = ax.scatter(z_normalized[front_mask, 0], z_normalized[front_mask, 1], z_normalized[front_mask, 2], 
                                c=subset_y[front_mask], cmap=cmap, norm=norm, s=10, alpha=1, marker=".", label='Front')
        ax.scatter(z_normalized[back_mask, 0], z_normalized[back_mask, 1], z_normalized[back_mask, 2], 
                c=subset_y[back_mask], cmap=cmap, norm=norm, s=10, alpha=0.2, marker=".", label='Back')
        
        # draw constraints
        if mergedCons is not None:
            px1 = X[mergedCons[:, 0]].float().to(device)
            px2 = X[mergedCons[:, 1]].float().to(device)
            type_values = mergedCons[:, 2]
            with torch.no_grad():
                h1 = self.encoder(px1)
                z1 = self._enc_mu(h1)
                z1_normalized = F.normalize(z1, p=2, dim=1).cpu().numpy()
                h2 = self.encoder(px2)
                z2 = self._enc_mu(h2)
                z2_normalized = F.normalize(z2, p=2, dim=1).cpu().numpy()
            num_constraints = min(num_cons, mergedCons.shape[0])
            for i in range(num_constraints):
                point1 = z1_normalized[i]
                point2 = z2_normalized[i]
                cons_type = type_values[i]
                if cons_type == 1:  
                    color = 'blue'
                    linestyle = '--'
                    alpha = 0.2
                else:  
                    color = 'red'
                    linestyle = '--'
                    alpha = 0.2
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]],
                        color=color, linestyle=linestyle, linewidth=1, alpha=alpha)
                
        # draw W vectors in assign stage
        if stage == "assign":
            for i in range(W_normalized.shape[0]):
                wx, wy, wz = W_normalized[i]
                ax.plot([0, wx], [0, wy], [0, wz], color='black', linewidth=2, linestyle='--')
                ax.scatter(wx, wy, wz, color='black', s=50, marker='o')
        
        ax.grid(False)
        ax.set_axis_off()
        # PNG
        if stage == "embedding":
            save_file = save_path.replace('.png', f'_embedding_epoch_{epoch}.png')
        elif stage == "assign":
            save_file = save_path.replace('.png', f'_assign.png')
        # PDF
        save_file = save_file.replace('.png', '.pdf')
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f'3D embeddings visualization saved to {save_file}')



    def fit(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2,
        lam, 
        X, y = None, 
        lr = 0.001, 
        batch_size = 256, 
        epochs = 500,
        soft_epochs = 100,
        tol = 0,
        omega = 2,
        plot_3D_path = None,
        record_feature_dir = None,
        record_feature_norm = False): 

        print("===================== Train a SDAE for clustering-friendly embedding space on Sphere ===================")
        optimizer_params = optim.Adam([param for param in self.parameters() if param is not self.W], lr=lr, weight_decay=0)
        
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long().to("cuda")
        mergedCons_num = mergedCons.shape[0]
        mergedCons_num_batch = int(math.ceil(1.0*mergedCons_num/batch_size))
        X_num = X.shape[0]
        X_batchsize = int(X_num/mergedCons_num_batch)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if y is not None:
            y = y.cpu().numpy()
        min_loss = 1e10
        last_loss = 1e10
        count_stop = 0

        self.train()


        for epoch in range(epochs):
            final_epoch = epoch
            """
            record log list:
                [   epoch,
                    loss_pairwise_train, loss_recon_train, loss_pair_recon_train, 
                    acc_train, acc_test, 
                    nmi_train, nmi_test,
                    ari_train, ari_test ]
            """
            list_log = [0,          # epoch
                        0,0,0,      # loss_pairwise_train, loss_recon_train, loss_pair_recon_train
                        0,0,        # acc_train, acc_test
                        0,0,        # nmi_train, nmi_test
                        0,0]        # ari_train, ari_test
            if record_feature_norm:
                 list_log = [0,          # epoch
                            0,0,0,      # loss_pairwise_train, loss_recon_train, loss_pair_recon_train
                            0,0,        # acc_train, acc_test
                            0,0,        # nmi_train, nmi_test
                            0,0,        # ari_train, ari_test
                            0,0]        # norm_avg, norm_std
            """ record log"""
            list_log[0] = epoch
            

            # ============================== 3D visualization ============================== #
            if plot_3D_path is not None:
                plot_3D_path = plot_3D_path
                if not os.path.exists(plot_3D_path):
                    os.makedirs(plot_3D_path)
                save_path = os.path.join(plot_3D_path, '3d_sphere.png')
                if epoch in [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]:
                    self.visualize_3d_sphere(X, y, epoch=epoch, train_ACC=None, num_samples=2000, num_cons=200, save_path=save_path, stage="embedding", mergedCons=mergedCons)


            # ============= lossY train, no batch, record log, check stop training condition, no update ============== #
            self.eval()
            with torch.no_grad(): 
                px1 = X[mergedCons[:, 0]]
                px2 = X[mergedCons[:, 1]]
                type_values = mergedCons[:, 2]
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                z1, _, _ = self.forward(inputs1)
                z2, _, _ = self.forward(inputs2)
                px = X
                inputs = Variable(px)
                _, xr, _ = self.forward(inputs)
                # losses
                loss_pairwise_train = self.pairwise_loss(z1, z2, type_values, omega)    
                loss_recon_train = self.recon_loss(inputs, xr)                          
                loss_pair_recon_train = loss_pairwise_train + lam * loss_recon_train    
                # logs
                list_log[1] = loss_pairwise_train.item()
                list_log[2] = loss_recon_train.item()
                list_log[3] = loss_pair_recon_train.item()  
                try:
                    delta_loss = loss_pair_recon_train.item() - min_loss 
                    delta = delta_loss/min_loss
                    if loss_pair_recon_train.item() < min_loss:
                        min_loss = loss_pair_recon_train.item()
                    if epoch > soft_epochs and delta < tol:
                        break
                except:
                    pass
                # # observed no difference
                # try:
                #     if abs(last_loss - loss_pair_recon_train.item())/last_loss < 0.1:
                #         count_stop += 1
                #     else:
                #         count_stop = 0
                #     if epoch > soft_epochs and count_stop >= 5:
                #         break
                # except:
                #     pass
            self.train()


            # ============= record norm ============= #
            if record_feature_norm:
                self.eval()
                with torch.no_grad(): 
                    inputs = Variable(X)
                    z, _, _ = self.forward(inputs)
                    z_norm = torch.norm(z, p=2, dim=1)
                    norm_avg = torch.mean(z_norm).item()
                    norm_std = torch.std(z_norm).item()
                    list_log[10] = norm_avg
                    list_log[11] = norm_std
                self.train()


            # ============================ update network =============================== #
            total_loss_pairwise = 0.0
            total_loss_recon = 0.0

            # shuffle mergedCons
            shuffled_indices = np.random.permutation(mergedCons.shape[0])
            mergedCons = mergedCons[shuffled_indices]
            shuffled_indices_X = np.random.permutation(X.shape[0])
            X_shuff = X[shuffled_indices_X]

            for cons_batch_idx in range(mergedCons_num_batch):
                px1 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 0]]
                px2 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 1]]
                type_values = mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 2]
                optimizer_params.zero_grad()
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                z1, _, _ = self.forward(inputs1)
                z2, _, _ = self.forward(inputs2)
                # pairwise loss
                loss_pairwise = self.pairwise_loss(z1, z2, type_values, omega)
                if torch.isnan(loss_pairwise):
                    print("loss_pairwise contains NaN")
                # recon loss
                px = X_shuff[cons_batch_idx*X_batchsize : min(X_num, (cons_batch_idx+1)*X_batchsize)]
                inputs = Variable(px)
                _, xr, _ = self.forward(inputs)
                loss_recon = self.recon_loss(inputs, xr)
                if torch.isnan(loss_recon):
                    print("loss_recon contains NaN")
                # total loss
                loss = loss_pairwise + lam * loss_recon        
                loss.backward()
                clip_grad_norm_(optimizer_params.param_groups[0]['params'], max_norm=1)
                optimizer_params.step()
                total_loss_pairwise += loss_pairwise.data * len(inputs1)
                total_loss_recon += loss_recon.data * len(inputs1)
            print("(Embedding) Epoch %d, Pairwise loss Total: %.5f, Recon loss Total: %.5f" % (epoch, float(total_loss_pairwise), float(total_loss_recon)))

                     
            # ================================ record log ================================ #
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
                writer.writerow(list_log)

        # ============================== save feature ============================== #
        if record_feature_dir is not None:
            latent = self.encodeBatch(X).cpu()
            torch.save(latent, record_feature_dir)
                
        return final_epoch
    


    #############################################################################
    #                            PCA-based K-inference                          #
    #############################################################################
    def analyze_K_with_dim_reduction(self, X, cl_ind1, cl_ind2, max_components=None, tail_ratio=0.05, sample_size=None, PCA=True):
        from lib.analyse_K_with_pca import analyze_K_with_pca_variance_core
        self.eval()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        z_norm = self.encodeBatch(X)
        z_np = z_norm.cpu().numpy()
        if PCA:
            method = 'PCA'
        else:
            method = 'TruncatedSVD'
        results = analyze_K_with_pca_variance_core(
            z_np, cl_ind1, cl_ind2, 
            max_components=max_components, 
            ratio=tail_ratio,
            sample_size=sample_size, 
            method=method
        )
        return results
    


    #############################################################################
    #                                assign_Kmeans                              #
    #############################################################################
    def assign_Kmeans(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, 
        X, y = None, 
        test_X = None, test_y = None,
        plot_3D_path = None,
        specific_K = None,
        n_init = 50):

        """
        record log list:
            [   epoch,
                loss_pairwise_train, loss_recon_train, loss_pair_recon_train,
                acc_train, acc_test, 
                nmi_train, nmi_test,
                ari_train, ari_test ]
        """
        list_log = [0,
                    0,0,0,
                    0,0,
                    0,0,
                    0,0]

        print("====================== Use Kmeans to find W for assignments ======================")
        # Check if specific_K is provided
        if specific_K is not None:
            self.n_clusters = specific_K
            self.W = Parameter(torch.Tensor(self.n_clusters, self.z_dim))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if y is not None:
            y = y.cpu().numpy()
        if test_y is not None:
            test_y = test_y.cpu().numpy()
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long().to("cuda")
        silhouette_score_avg = None  # Initialize silhouette score value


        if specific_K is not None:
            # ============================ try different K ============================ #
            print(f"Finding cluster center W with Kmeans for specific_K={specific_K}")
            data = self.encodeBatch(X)
            data_np = data.data.cpu().numpy()
            
            sample_size = 5000
            W = self.W
            best_kmeans = None
            best_inertia = float('inf')
            silhouette_scores = []
            sc_n_init = 5
            for i in range(sc_n_init):
                kmeans = KMeans(self.n_clusters, n_init=1)
                y_pred = kmeans.fit_predict(data_np)
                score = silhouette_score(data_np, y_pred, sample_size=sample_size)
                silhouette_scores.append(score)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
                print(f"Kmeans run {i+1}/{sc_n_init} for specific_K={specific_K}, inertia: {kmeans.inertia_}, silhouette score: {score:.5f}")
            average_score = np.mean(silhouette_scores) 
            silhouette_score_avg = average_score
            silhouette_score_std = np.std(silhouette_scores)
            print(f"Average Silhouette Score for specific_K={specific_K}: {silhouette_score_avg:.5f}")
            W.data.copy_(torch.Tensor(best_kmeans.cluster_centers_))
        else:
            # ============================= ground truth K ============================ #
            print("Finding cluster center W with Kmeans")
            data = self.encodeBatch(X)
            data_np = data.data.cpu().numpy()
            W = self.W
            kmeans = KMeans(self.n_clusters, n_init=n_init)   
            y_pred = kmeans.fit_predict(data_np)
            W.data.copy_(torch.Tensor(kmeans.cluster_centers_)) 


        # ============================ Predicting with W ============================ #
        # train set
        latent = self.encodeBatch(X)
        W = self.W
        _, y_pred = self.hard_assign(latent, W)
        y_pred = y_pred.data.cpu().numpy()
        satisfied_cons = self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred)

        # test set
        if test_X is not None:
            test_latent = self.encodeBatch(test_X)
            W = self.W
            test_y_pred = self.hard_assign(test_latent, W)[1]
            test_y_pred = test_y_pred.data.cpu().numpy()

        
        # =============================== record performance ============================== #
        # train set
        if y is not None:
            print("acc: %.5f, nmi: %.5f, ari:%.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
            print("satisfied constraints: %.5f" % satisfied_cons)
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)
            """record log"""
            list_log[4] = final_acc
            list_log[6] = final_nmi
            list_log[8] = final_ari
        
        # test set
        if test_X is not None and test_y is not None:
            test_acc = acc(test_y, test_y_pred)
            test_nmi = normalized_mutual_info_score(test_y, test_y_pred)
            test_ari = adjusted_rand_score(test_y, test_y_pred)
            """record log"""
            list_log[5] = test_acc
            list_log[7] = test_nmi
            list_log[9] = test_ari


        # ============================== 3D visual ============================== #
        if plot_3D_path is not None:
            plot_3D_path = plot_3D_path
            if not os.path.exists(plot_3D_path):
                os.makedirs(plot_3D_path)
            save_path = os.path.join(plot_3D_path, '3d_sphere.png')
            self.visualize_3d_sphere(X, y, epoch=None, train_ACC=final_acc, num_samples=3000, num_cons=400, save_path=save_path, stage="assign", mergedCons=mergedCons)
        

        # ================================ record log ================================ #
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
    



    #############################################################################
    #                          assign_Hierarchical                              #
    #############################################################################
    def assign_Hierarchical(
        self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, 
        X, y=None, 
        test_X=None, test_y=None,
        plot_3D_path=None,
        specific_K=None,
        method='ward'     # You can change to other linkage methods, such as 'single', 'complete', 'average', etc.
    ):
        # Log list: [epoch, loss_pairwise_train, loss_recon_train, loss_pair_recon_train,
        #           acc_train, acc_test, nmi_train, nmi_test, ari_train, ari_test]
        list_log = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        print("====================== Use Hierarchical Clustering to find W for assignments ======================")
        
        # If specific_K is specified, override self.n_clusters
        if specific_K is not None:
            self.n_clusters = specific_K
            self.W = Parameter(torch.Tensor(self.n_clusters, self.z_dim))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        # If y/test_y is not None, convert to numpy for subsequent evaluation
        if y is not None:
            y = y.cpu().numpy()
        if test_y is not None:
            test_y = test_y.cpu().numpy()
        
        # Only used to record the proportion of satisfied constraints
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2)).long()
        if use_cuda:
            mergedCons = mergedCons.cuda()
        
        silhouette_score_value = None  # Initialize silhouette score
        
        # ============== 1) Encode the training set to obtain low-dimensional representations ==============
        data = self.encodeBatch(X)
        data_np = data.data.cpu().numpy()

        # ============== 2) Use hierarchical clustering to obtain cluster labels ==============
        print(f"Finding cluster center W with Hierarchical Clustering (method='{method}')...")
        
        # Calculate linkage matrix
        Z = fastcluster.linkage(data_np, method=method)
        # Specify to divide into self.n_clusters clusters
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')  # labels range from 1 to self.n_clusters
        
        # ============== 3) Calculate cluster centers and update model parameter W ==============
        centers = np.zeros((self.n_clusters, data_np.shape[1]))
        for c in range(self.n_clusters):
            cluster_points = data_np[labels == (c + 1)]
            # If there are empty clusters, simple handling can be done; here it is assumed that empty clusters do not occur or can be skipped
            if len(cluster_points) > 0:
                centers[c] = cluster_points.mean(axis=0)
            else:
                centers[c] = 0  # Or randomly initialize a center
        
        self.W.data.copy_(torch.Tensor(centers))  # Update network parameters
        
        # If specific_K is specified, calculate Silhouette Score as a simple reference
        if specific_K is not None:
            sample_size = min(5000, len(data_np))  # Use up to 5000 samples to compute silhouette
            silhouette_score_value = silhouette_score(data_np, labels, sample_size=sample_size, random_state=42)
            print(f"Silhouette Score for Hierarchical Clustering (K={specific_K}): {silhouette_score_value:.5f}")

        # ============== 4) Predict on the training set, calculate metrics + constraint satisfaction ==============
        # Use Partition S (cluster labels directly generated by hierarchical clustering) to calculate training set metrics
        final_acc, final_nmi, final_ari = 0, 0, 0
        if y is not None:
            # Convert labels (1~K) to 0~K-1
            labels_zero_based = labels - 1
            final_acc = acc(y, labels_zero_based)
            final_nmi = normalized_mutual_info_score(y, labels_zero_based)
            final_ari = adjusted_rand_score(y, labels_zero_based)
            satisfied_cons = self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, labels_zero_based)
            print("acc: %.5f, nmi: %.5f, ari: %.5f" % (final_acc, final_nmi, final_ari))
            print("satisfied constraints: %.5f" % satisfied_cons)
            list_log[4] = final_acc
            list_log[6] = final_nmi
            list_log[8] = final_ari

        # ============== 5) Predict on the test set (if provided) ==============
        test_acc, test_nmi, test_ari = 0, 0, 0
        if test_X is not None and test_y is not None:
            # Assign test set samples using centroids W
            test_latent = self.encodeBatch(test_X)
            _, test_y_pred = self.hard_assign(test_latent, self.W)
            test_y_pred = test_y_pred.data.cpu().numpy()
            test_acc = acc(test_y, test_y_pred)
            test_nmi = normalized_mutual_info_score(test_y, test_y_pred)
            test_ari = adjusted_rand_score(test_y, test_y_pred)
            list_log[5] = test_acc
            list_log[7] = test_nmi
            list_log[9] = test_ari

        # ============== 6) Optional 3D visualization ==============
        if plot_3D_path is not None:
            if not os.path.exists(plot_3D_path):
                os.makedirs(plot_3D_path)
            save_path = os.path.join(plot_3D_path, '3d_sphere.png')
            # Note: self.z_dim must be 3 to plot a 3D sphere
            self.visualize_3d_sphere(X, y, epoch=None, train_ACC=final_acc, num_samples=2000,
                                    num_cons=200, save_path=save_path, stage="assign", mergedCons=mergedCons)
        
        # ============== 7) Log records to CSV ==============
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

        # ============== 8) Return results ==============
        if specific_K is not None:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari, silhouette_score_value
        else:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari

