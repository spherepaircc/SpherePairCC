import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import csv
from lib.utils import acc, delta_label_sum
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


class VolMaxDCC(nn.Module):
    def __init__(self, input_dim, n_clusters, hidden_layers, activation="relu", is_B_trainable=True, B_init=None,):
        super(VolMaxDCC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.buildNetwork(input_dim, n_clusters, hidden_layers, activation)

        self.is_B_trainable = is_B_trainable
        const = 1.0
        if is_B_trainable:
            B_init = const*torch.eye(n_clusters)
            B_init[B_init==0] = -const
            self._B = nn.Parameter(B_init, requires_grad=True)
            self._sigmoid = nn.Sigmoid()
        else:
            if B_init is None:
                self._B = torch.eye(n_clusters).to(self.device)
            else:
                self._B = torch.from_numpy(B_init.astype(np.float32)).to(self.device)
            self._sigmoid = lambda x: x

        self.to(self.device)
    

    def _get_B(self):
        return self._sigmoid(self._B)


    def buildNetwork(self, input_dim, n_clusters, hidden_layers, activation="relu"):
        layers_list = []
        layers = [input_dim] + hidden_layers + [n_clusters]
        for i in range(1, len(layers)):
            layers_list.append(nn.Linear(layers[i-1], layers[i]))
            if i < len(layers) - 1: 
                layers_list.append(nn.BatchNorm1d(layers[i]))
                if activation == "relu":
                    layers_list.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers_list.append(nn.Sigmoid())
                layers_list.append(nn.Dropout(0))
        layers_list.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers_list)


    def soft_assign(self, X):
        X = X.to(self.device)
        output = self.net(X)
        return output


    def predict(self, q):
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        return y_pred
    

    def evaluate(self, y, y_pred):
        the_acc = acc(y, y_pred)
        the_nmi = normalized_mutual_info_score(y, y_pred)
        the_ari = adjusted_rand_score(y, y_pred)
        return the_acc, the_nmi, the_ari


    def compute_loss_1(self, B, q1, q2, type_values):
        link_pred_probs = torch.mul(torch.matmul(q1, B), q2).sum(-1)
        epsilon = 1e-6
        link_pred_probs = torch.clamp(link_pred_probs, min=epsilon, max=1-epsilon)    
        loss_1 = F.binary_cross_entropy(link_pred_probs, type_values.float(), reduction='mean')
        return loss_1
    

    # A larger loss_2 represents greater volume/diversity/linear independence/spread of F
    # Reducing loss_1 means increasing loss_2 (L = loss1 - lambda*loss2)
    def compute_loss_2(self, F):   
        cov = torch.matmul(F.T, F)
        det_cov = torch.det(cov)
        if det_cov <= 0:
            det_cov = torch.tensor(1e-6, device=self.device)
        return torch.log(det_cov)
    

    # Use a subset X_sub to compute loss_2 for faster computation
    # Using the full X would make F's computation prohibitively large
    # (The original author uniformly used only 10K samples 
    # as the train set for any dataset, 
    # leaving the rest as the test set)
    def compute_loss(self, B, q1, q2, type_values, lam, X):
        loss_1 = self.compute_loss_1(B, q1, q2, type_values)
        if self.is_B_trainable:
            N = X.size(0)
            N_prime = 10000  # subset size
            indices = torch.randperm(N)[:N_prime]
            X_sub = X[indices, :]
            F = self.soft_assign(X_sub)
            loss_2 = self.compute_loss_2(F)
            loss = loss_1 - lam * loss_2
            return loss, loss_1, loss_2
        else:
            return loss_1, loss_1, loss_1


    def satisfied_constraints(self, mergedConstraints, y_pred):
        if mergedConstraints.size == 0:
            return 1.1
        count = 0
        satisfied = 0
        for constraint in mergedConstraints:
            i, j, label = constraint
            if label == 1:
                count += 1
                if y_pred[int(i)] == y_pred[int(j)]:
                    satisfied += 1
            elif label == 0:
                count += 1
                if y_pred[int(i)] != y_pred[int(j)]:
                    satisfied += 1
        return float(satisfied) / count if count > 0 else 0
    

    def merge_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
        ml_labels = np.ones(len(ml_ind1))
        cl_labels = np.zeros(len(cl_ind1))
        ml_cons = np.column_stack((ml_ind1, ml_ind2, ml_labels))
        cl_cons = np.column_stack((cl_ind1, cl_ind2, cl_labels))
        mergedCons = np.concatenate((ml_cons, cl_cons), axis=0)
        np.random.shuffle(mergedCons)
        return mergedCons


    def fit(self, 
        record_log_dir,
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, 
        val_ml_ind1, val_ml_ind2, val_cl_ind1, val_cl_ind2,  
        X, y = None, 
        test_X = None, test_y = None, 
        lr = 0.001, 
        lam = 0.01,
        batch_size = 256, 
        num_epochs = 500, 
        tol = 1e-3):

        print("=================================Training VolMaxDCC ===================================")
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)   # Adam, a bit more stable than SGD, though the author used SGD in paper
        # SGD
        optimizer = torch.optim.SGD([
            {'params': [param for name, param in self.named_parameters() if name != '_B'], 'lr': lr},
            {'params': [self._B], 'lr': 0.1}
        ], weight_decay=0)


        y = y.cpu().numpy()
        test_y = test_y.cpu().numpy()
        num = X.shape[0]
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long().to("cuda")
        mergedCons_num = mergedCons.shape[0]
        mergedCons_num_batch = int(math.ceil(1.0*mergedCons_num/batch_size))
        if val_ml_ind1 is not None:
            val_mergedCons = torch.from_numpy(self.merge_constraints(val_ml_ind1, val_ml_ind2, val_cl_ind1, val_cl_ind2))
            val_mergedCons = val_mergedCons.long().to("cuda")

        final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari = 0, 0, 0, 0, 0, 0, 0

        self.train()


        for epoch in range(num_epochs):
            final_epoch = epoch
            """
            record log list:
                [  epoch,
                    lossY_train, lossY_test, 
                    acc_train, acc_test, 
                    nmi_train, nmi_test,
                    ari_train, ari_test ]
            """
            list_log = [0,          # epoch
                        0,0,        # lossY_train, lossY_test
                        0,0,        # acc_train, acc_test
                        0,0,        # nmi_train, nmi_test
                        0,0]        # ari_train, ari_test
            """record log"""
            list_log[0] = epoch


            # ===================== updating, recording, and checking the stopping condition at each epoch ========================
            self.eval()
            q = self.soft_assign(X)
            y_pred = self.predict(q)

            # train performance
            if y is not None:
                final_acc, final_nmi, final_ari = self.evaluate(y, y_pred)
                print(f"acc: {final_acc:.5f}, nmi: {final_nmi:.5f}, ari: {final_ari:.5f}")
                print("satisfied constraints: %.5f"%self.satisfied_constraints(mergedCons, y_pred))
                """record log"""
                list_log[3] = final_acc
                list_log[5] = final_nmi
                list_log[7] = final_ari
            
            # test performance
            if test_X is not None and test_y is not None:
                q_test = self.soft_assign(test_X)
                y_pred_test = self.predict(q_test)
                test_acc, test_nmi, test_ari = self.evaluate(test_y, y_pred_test)
                """record log"""
                list_log[4] = test_acc
                list_log[6] = test_nmi
                list_log[8] = test_ari
            self.train()
            
            # stop
            if epoch == 0:
                y_pred_last = y_pred
                delta_label = delta_label_sum(y_pred, y_pred_last) / num
            else:
                delta_label = delta_label_sum(y_pred, y_pred_last) / num
                y_pred_last = y_pred
            if epoch>10 and delta_label < tol:   # dont stop too early for stable
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                break
            else:
                print('delta_label ', delta_label, '>= tol ', tol)


            # whether to record the detail
            RECORD_DETAILS_FLAG = False


            if RECORD_DETAILS_FLAG == True:
                # ============= lossY_train, no batch, record log, no update ============== #
                self.eval()
                with torch.no_grad():
                    px1 = X[mergedCons[:, 0]]
                    px2 = X[mergedCons[:, 1]]
                    type_values = mergedCons[:, 2]
                    q1 = self.soft_assign(px1)
                    q2 = self.soft_assign(px2)
                    lossY_train, _, _ = self.compute_loss(self._get_B(), q1, q2, type_values, lam, X).item()
                    """record log"""
                    list_log[1] = lossY_train
                self.train()

                # ============= lossY_test, no batch, record log, no update ================ #
                if test_X is not None and val_ml_ind1 is not None:
                    self.eval()
                    with torch.no_grad():
                        test_px1 = test_X[val_mergedCons[:, 0]]
                        test_px2 = test_X[val_mergedCons[:, 1]]
                        test_type_values = val_mergedCons[:, 2]
                        test_q1 = self.soft_assign(test_px1)
                        test_q2 = self.soft_assign(test_px2)
                        lossY_test, _, _ = self.compute_loss(self._get_B(), test_q1, test_q2, test_type_values, lam, test_X).item()
                        """record log"""
                        list_log[2] = lossY_test
                    self.train()
            
            # ================== lossY_train, update ==================== #
            total_pairwise_loss = 0.0

            # shuffle
            temp_indices = np.random.permutation(np.arange(mergedCons.shape[0]))
            temp_indices = torch.from_numpy(temp_indices)
            reverse_temp_indices = torch.argsort(temp_indices.clone().detach())
            mergedCons = mergedCons[temp_indices]

            for cons_batch_idx in range(mergedCons_num_batch):
                px1 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 0]]
                px2 = X[mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 1]]
                type_values = mergedCons[cons_batch_idx*batch_size : min(mergedCons_num, (cons_batch_idx+1)*batch_size), 2]
                q1 = self.soft_assign(px1)
                q2 = self.soft_assign(px2)
                lossY, _, _ = self.compute_loss(self._get_B(), q1, q2, type_values, lam, X)
                total_pairwise_loss += lossY * len(px1) 
                optimizer.zero_grad()
                lossY.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
            total_pairwise_loss = total_pairwise_loss / mergedCons_num
            print("Pairwise Total:", float(total_pairwise_loss.cpu()))

            mergedCons = mergedCons[reverse_temp_indices]
            

            # ================================ record log ================================ #
            if not os.path.exists(record_log_dir):
                with open(record_log_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["epoch", 
                                     "lossY_train", "lossY_test",
                                     "acc_train", "acc_test",
                                     "nmi_train", "nmi_test",
                                     "ari_train", "ari_test"])
            with open(record_log_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list_log)
        


        return final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari