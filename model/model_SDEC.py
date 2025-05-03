import torch
import torch.nn as nn
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
from torch.nn.utils import clip_grad_norm_



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


class SDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1.):
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
        self.alpha = alpha
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))


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
        z = self._enc_mu(h)   
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon


    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


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
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded


    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        kldloss = kld(p, q)
        return kldloss
    

    def pairwise_loss(self, z1, z2, type_values):
        loss_ml = torch.sum(type_values * torch.sum((z1 - z2)**2, dim=1))
        loss_cl = torch.sum((1 - type_values) * torch.sum((z1 - z2)**2, dim=1))
        n_num = z1.size(0)
        return (loss_ml - loss_cl) / n_num


    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p


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
        q = self.soft_assign(latent)
        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
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


    def fit(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, 
        lam, 
        X, y = None, 
        test_X = None, test_y = None, 
        lr = 0.001, 
        batch_size = 256, 
        num_epochs = 500, 
        tol = 1e-3, 
        use_kmeans = True,
        record_feature_dir = None): 

        print("=================================Training SDEC===================================")
        # Adam
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # SGD
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long().to("cuda")
        mergedCons_num = mergedCons.shape[0]
        X_num = X.shape[0]

        mergedCons_num_batch = int(math.ceil(1.0*mergedCons_num/batch_size))
        X_batchsize = int(X_num/mergedCons_num_batch)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if y is not None:
            y = y.cpu().numpy()
        
        self.train() 
        
        if use_kmeans:
            print("Initializing cluster centers with kmeans.")
            kmeans = KMeans(self.n_clusters, n_init=20)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        else:
            # use kmeans to randomly initialize cluster ceters
            print("Randomly initializing cluster centers.")
            kmeans = KMeans(self.n_clusters, n_init=1, max_iter=1)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari = 0, 0, 0, 0, 0, 0, 0

        self.train()


        for epoch in range(num_epochs):
            final_epoch = epoch
            """
            record log list:
                [  epoch,
                    lossX_train, lossX_test, 
                    lossY_train, lossY_test, 
                    acc_train, acc_test, 
                    nmi_train, nmi_test,
                    ari_train, ari_test ]
            """
            list_log = [0,       # epoch
                        0,0,     # lossX_train, lossX_test
                        0,0,     # lossY_train, lossY_test
                        0,0,     # acc_train, acc_test
                        0,0,     # nmi_train, nmi_test
                        0,0]     # ari_train, ari_test
            """ record log """
            list_log[0] = epoch
            

            # ===================== updating, recording, and checking the stopping condition at each epoch ========================
            if epoch%1 == 0:

                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # train performance
                if y is not None:
                    print("acc: %.5f, nmi: %.5f, ari:%.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
                    print("satisfied constraints: %.5f"%self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred))
                    final_acc = acc(y, y_pred)
                    final_nmi = normalized_mutual_info_score(y, y_pred)
                    final_ari = adjusted_rand_score(y, y_pred)
                    """ record log """
                    list_log[5] = final_acc
                    list_log[7] = final_nmi
                    list_log[9] = final_ari
                
                # test performance
                if test_X is not None and test_y is not None:
                    test_acc, test_nmi, test_ari = self.predict(test_X, test_y)
                    """ record log """
                    list_log[6] = test_acc
                    list_log[8] = test_nmi
                    list_log[10] = test_ari

                # stop
                try:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X_num
                    y_pred_last = y_pred
                    if epoch>0 and delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                except:
                    pass
            

            # ================== lossX + lam * lossY, update ==================== #
            # shuffle mergedCons
            shuffled_indices = np.random.permutation(mergedCons.shape[0])
            mergedCons = mergedCons[shuffled_indices]

            # X_shuff in each epoch
            shuffled_indices_X = np.random.permutation(X.shape[0])
            X_shuff = X[shuffled_indices_X]
            p_shuff = p[shuffled_indices_X]

            # pairwise_loss and cluster_loss
            for cons_batch_idx in range(mergedCons_num_batch):
                
                # pairwise_loss
                px1 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 0]]
                px2 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 1]]
                type_values = mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 2]
                optimizer.zero_grad()
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                z1, _, _ = self.forward(inputs1)
                z2, _, _ = self.forward(inputs2)
                lossY = self.pairwise_loss(z1, z2, type_values)

                # cluster_loss
                xbatch = X_shuff[cons_batch_idx*X_batchsize : min(X_num, (cons_batch_idx+1)*X_batchsize)]
                pbatch = p_shuff[cons_batch_idx*X_batchsize : min(X_num, (cons_batch_idx+1)*X_batchsize)]
                inputs = Variable(xbatch)
                target = Variable(pbatch)
                _, qbatch, _ = self.forward(inputs)
                lossX = self.cluster_loss(target, qbatch)

                # total loss
                loss = lossX + lam * lossY

                loss.backward()
                # clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()

             
            # ================================ log ================================ #
            if not os.path.exists(record_log_dir): 
                with open(record_log_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["epoch", 
                                     "lossX_train", "lossX_test", 
                                     "lossY_train", "lossY_test", 
                                     "acc_train", "acc_test", 
                                     "nmi_train", "nmi_test",
                                     "ari_train", "ari_test"])
            with open(record_log_dir, "a") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(list_log)
        

        # ============================== save feature ============================== #
        if record_feature_dir is not None:
            print("Saving the learned features.")
            latent = self.encodeBatch(X).cpu()
            torch.save(latent, record_feature_dir)
            

        return final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari
