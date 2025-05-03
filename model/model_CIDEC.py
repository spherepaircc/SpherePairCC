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


class CIDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
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
        self.gamma = gamma
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
        return self.gamma*kldloss


    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon-x)**2)
        return recon_loss


    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss


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


    def fit(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, 
        val_ml_ind1, val_ml_ind2, val_cl_ind1, val_cl_ind2,  
        ml_p, cl_p, 
        X, y = None, 
        test_X = None, test_y = None, 
        lr = 0.001, 
        batch_size = 256, 
        num_epochs = 500, 
        tol = 1e-3, 
        use_kmeans = True,
        clustering_loss_weight = 1,
        record_feature_dir = None): 

        original_ml_ind1 = ml_ind1.copy()
        original_ml_ind2 = ml_ind2.copy()
        original_cl_ind1 = cl_ind1.copy()
        original_cl_ind2 = cl_ind2.copy()   
        
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=================================Training CIDEC===================================")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

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

        if y is not None:
            y = y.cpu().numpy()

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        ml_num = ml_ind1.shape[0]
        cl_num = cl_ind1.shape[0]

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
            # update the targe distribution p
            # on the train set
            latent = self.encodeBatch(X)
            q = self.soft_assign(latent)
            p = self.target_distribution(q).data
            # on the test set
            if test_X is not None:
                test_latent = self.encodeBatch(test_X)
                test_q = self.soft_assign(test_latent)
                test_p = self.target_distribution(test_q).data

            # evalute the clustering performance
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

            # train set performance
            if y is not None:
                print("acc: %.5f, nmi: %.5f, ari:%.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
                print("satisfied constraints: %.5f"%self.satisfied_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2, y_pred))
                final_acc = acc(y, y_pred)
                final_nmi = normalized_mutual_info_score(y, y_pred)
                final_ari = adjusted_rand_score(y, y_pred)
                """record log"""
                list_log[5] = final_acc
                list_log[7] = final_nmi
                list_log[9] = final_ari
            
            # test set performance
            if test_X is not None and test_y is not None:
                test_acc, test_nmi, test_ari = self.predict(test_X, test_y)
                """ record log """
                list_log[6] = test_acc
                list_log[8] = test_nmi
                list_log[10] = test_ari

            # stop
            try:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                # if epoch>0 and delta_label < tol:
                if epoch>30 and delta_label < tol:  # dont stop in the first 30 epochs for
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
            except:
                pass
            
            # whether to record the detailed training log
            RECORD_DETAIL_FLAG = False

            if RECORD_DETAIL_FLAG == True:
                # ============= lossX_train, no batch, record log, no update ============== #
                self.eval()
                with torch.no_grad(): 
                    train_inputs = Variable(X)
                    train_target = Variable(p)
                    _, train_q, train_xrecon = self.forward(train_inputs)
                    train_cluster_loss = self.cluster_loss(train_target, train_q)
                    train_recon_loss = self.recon_loss(train_inputs, train_xrecon)
                    train_lossX = clustering_loss_weight * train_cluster_loss + train_recon_loss
                    """ record log """
                    list_log[1] = train_lossX.item()
                self.train()

                # ============= lossX_test, no batch, record log, no update =============== #
                if test_X is not None:
                    self.eval()
                    with torch.no_grad(): 
                        test_inputs = Variable(test_X)
                        test_target = Variable(test_p)
                        _, test_q, test_xrecon = self.forward(test_inputs)
                        test_cluster_loss = self.cluster_loss(test_target, test_q)
                        test_recon_loss = self.recon_loss(test_inputs, test_xrecon)
                        test_lossX = clustering_loss_weight * test_cluster_loss + test_recon_loss
                        """ record log """
                        list_log[2] = test_lossX.item()
                    self.train()

            # ================== lossX_train, update ==================== #
            lossX_train_allBatch = 0.0
            cluster_loss_allBatch = 0.0
            recon_loss_allBatch = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                # forward
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)
                _, qbatch, xrecon = self.forward(inputs)
                # clustering loss + reconstruction loss
                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.recon_loss(inputs, xrecon)
                loss = clustering_loss_weight * cluster_loss + recon_loss
                # backward
                loss.backward()
                optimizer.step()
                # total loss
                cluster_loss_allBatch += cluster_loss.data * len(inputs)
                recon_loss_allBatch += recon_loss.data * len(inputs)
                lossX_train_allBatch = clustering_loss_weight * cluster_loss_allBatch + recon_loss_allBatch
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                epoch + 1, lossX_train_allBatch / num, cluster_loss_allBatch / num, recon_loss_allBatch / num))            


            if RECORD_DETAIL_FLAG == True:
                # ============= lossY_train, no batch, record log, no update ============== #
                self.eval()
                with torch.no_grad():
                    # ML loss
                    px1 = X[ml_ind1]
                    px2 = X[ml_ind2]
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    _, q1, xr1 = self.forward(inputs1)
                    _, q2, xr2 = self.forward(inputs2)
                    train_ml_loss = (ml_p * self.pairwise_loss(q1, q2, "ML") + self.recon_loss(inputs1, xr1) + self.recon_loss(inputs2, xr2))
                    # CL loss
                    px1 = X[cl_ind1]
                    px2 = X[cl_ind2]
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    _, q1, xr1 = self.forward(inputs1)
                    _, q2, xr2 = self.forward(inputs2)
                    train_cl_loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    # Total lossY_train
                    lossY_train = (train_ml_loss + train_cl_loss).item()
                    """ record log """
                    list_log[3] = lossY_train
                self.train()
                
                # ============= lossY_test, no batch, record log, no update ================ #
                if test_X is not None and val_ml_ind1 is not None:
                    self.eval()
                    with torch.no_grad():
                        # ML loss
                        test_px1 = test_X[val_ml_ind1]
                        test_px2 = test_X[val_ml_ind2]
                        test_inputs1 = Variable(test_px1)
                        test_inputs2 = Variable(test_px2)
                        _, test_q1, test_xr1 = self.forward(test_inputs1)
                        _, test_q2, test_xr2 = self.forward(test_inputs2)
                        test_ml_loss = (ml_p * self.pairwise_loss(test_q1, test_q2, "ML") + self.recon_loss(test_inputs1, test_xr1) + self.recon_loss(test_inputs2, test_xr2))
                        # CL loss
                        test_px1 = test_X[val_cl_ind1]
                        test_px2 = test_X[val_cl_ind2]
                        test_inputs1 = Variable(test_px1)
                        test_inputs2 = Variable(test_px2)
                        _, test_q1, test_xr1 = self.forward(test_inputs1)
                        _, test_q2, test_xr2 = self.forward(test_inputs2)
                        test_cl_loss = cl_p * self.pairwise_loss(test_q1, test_q2, "CL")
                        # Total lossY_test
                        lossY_test = test_ml_loss + test_cl_loss
                        """ record log """
                        list_log[4] = lossY_test.item()
                    self.train()

            # ================== lossY_train, update ==================== #
            # ML loss
            ml_loss = 0.0
            # shuffle
            permuted_indices = np.random.permutation(len(ml_ind1))
            ml_ind1 = ml_ind1[permuted_indices]
            ml_ind2 = ml_ind2[permuted_indices]
            for ml_batch_idx in range(ml_num_batch):
                px1 = X[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                px2 = X[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                # forward
                optimizer.zero_grad()
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                _, q1, xr1 = self.forward(inputs1)
                _, q2, xr2 = self.forward(inputs2)
                # ML loss
                loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.recon_loss(inputs1, xr1) + self.recon_loss(inputs2, xr2))
                ml_loss += loss.data * len(inputs1)
                # backward
                loss.backward()
                optimizer.step()
            ml_loss = ml_loss / ml_num
            # restore the original order
            ml_ind1 = original_ml_ind1.copy()
            ml_ind2 = original_ml_ind2.copy()

            # CL loss
            cl_loss = 0.0
            # shuffle
            permuted_indices = np.random.permutation(len(cl_ind1))
            cl_ind1 = cl_ind1[permuted_indices]
            cl_ind2 = cl_ind2[permuted_indices]
            for cl_batch_idx in range(cl_num_batch):
                px1 = X[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                px2 = X[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                # forward
                optimizer.zero_grad()
                inputs1 = Variable(px1)
                inputs2 = Variable(px2)
                _, q1, xr1 = self.forward(inputs1)
                _, q2, xr2 = self.forward(inputs2)
                # CL loss
                loss = cl_p*self.pairwise_loss(q1, q2, "CL")
                cl_loss += loss.data * len(inputs1) 
                # backward
                loss.backward()
                optimizer.step()
            cl_loss = cl_loss / cl_num 
            # restore the original order
            cl_ind1 = original_cl_ind1.copy()
            cl_ind2 = original_cl_ind2.copy()

            print("Pairwise Total:", float(ml_loss) + float(cl_loss), "ML loss", float(ml_loss), "CL loss:", float(cl_loss))
            
            
                 
            # ================================ record log ================================ #
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
