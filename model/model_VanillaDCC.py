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

class VanillaDCC(nn.Module):

    def __init__(self, input_dim, n_clusters, hidden_layers, activation="relu"):
        super(VanillaDCC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.buildNetwork(input_dim, n_clusters, hidden_layers, activation)
        self.to(self.device)

    def buildNetwork(self, input_dim, n_clusters, hidden_layers, activation="relu"):
        feature_layers = []
        layer_sizes = [input_dim] + hidden_layers
        for i in range(1, len(layer_sizes)):
            feature_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            feature_layers.append(nn.BatchNorm1d(layer_sizes[i]))
            if activation == "relu":
                feature_layers.append(nn.ReLU())
            elif activation == "sigmoid":
                feature_layers.append(nn.Sigmoid())
            feature_layers.append(nn.Dropout(0))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.cluster_layer = nn.Sequential(
            nn.Linear(hidden_layers[-1], n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        features = self.feature_extractor(X)
        cluster_probs = self.cluster_layer(features)
        return cluster_probs

    def soft_assign(self, X):
        X = X.to(self.device)
        output = self.forward(X)
        return output

    # get output of the last hidden layer
    def get_features(self, X):
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(X.to(self.device))
        return features.cpu()

    # soft assignments
    def predict(self, q):
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        return y_pred

    def evaluate(self, y, y_pred):
        the_acc = acc(y, y_pred)
        the_nmi = normalized_mutual_info_score(y, y_pred)
        the_ari = adjusted_rand_score(y, y_pred)
        return the_acc, the_nmi, the_ari

    def pairwise_loss(self, q1, q2, type_values):
        pred_probs = torch.sum(q1 * q2, dim=1)
        loss = F.binary_cross_entropy(pred_probs, type_values.float(), reduction='mean')
        return loss

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
            batch_size = 256, 
            num_epochs = 500, 
            tol = 1e-3,
            record_feature_dir = None): 

        print("=================================Training VanillaDCC ===================================")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)   # Adam
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0)      # SGD

        if y is not None:
            y = y.cpu().numpy()
        if test_y is not None:
            test_y = test_y.cpu().numpy()

        num = X.shape[0]
        mergedCons = torch.from_numpy(self.merge_constraints(ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        mergedCons = mergedCons.long().to(self.device)
        mergedCons_num = mergedCons.shape[0]
        mergedCons_num_batch = int(math.ceil(1.0 * mergedCons_num / batch_size))

        if val_ml_ind1 is not None:
            val_mergedCons = torch.from_numpy(self.merge_constraints(val_ml_ind1, val_ml_ind2, val_cl_ind1, val_cl_ind2))
            val_mergedCons = val_mergedCons.long().to(self.device)

        final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari = 0, 0, 0, 0, 0, 0, 0

        self.train()

        for epoch in range(num_epochs):
            final_epoch = epoch
            list_log = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            list_log[0] = epoch

            self.eval()
            q = self.soft_assign(X)
            y_pred = self.predict(q)

            if y is not None:
                final_acc, final_nmi, final_ari = self.evaluate(y, y_pred)
                print(f"Epoch {epoch}: acc: {final_acc:.5f}, nmi: {final_nmi:.5f}, ari: {final_ari:.5f}")
                print("Satisfied constraints: %.5f" % self.satisfied_constraints(mergedCons, y_pred))
                list_log[3] = final_acc
                list_log[5] = final_nmi
                list_log[7] = final_ari

            if test_X is not None and test_y is not None:
                q_test = self.soft_assign(test_X)
                y_pred_test = self.predict(q_test)
                test_acc, test_nmi, test_ari = self.evaluate(test_y, y_pred_test)
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
            if epoch > 10 and delta_label < tol:
                print('Delta_label {:.6f} < tol {:.6f}'.format(delta_label, tol))
                print("Reach tolerance threshold. Stopping training.")
                break
            else:
                print('Delta_label {:.6f} >= tol {:.6f}'.format(delta_label, tol))

            RECORD_DETAILS_FLAG = False

            if RECORD_DETAILS_FLAG:
                self.eval()
                with torch.no_grad():
                    px1 = X[mergedCons[:, 0]]
                    px2 = X[mergedCons[:, 1]]
                    type_values = mergedCons[:, 2]
                    q1 = self.soft_assign(px1)
                    q2 = self.soft_assign(px2)
                    lossY_train = self.pairwise_loss(q1, q2, type_values).item()
                    list_log[1] = lossY_train
                self.train()

                if test_X is not None and val_ml_ind1 is not None:
                    self.eval()
                    with torch.no_grad():
                        test_px1 = test_X[val_mergedCons[:, 0]]
                        test_px2 = test_X[val_mergedCons[:, 1]]
                        test_type_values = val_mergedCons[:, 2]
                        test_q1 = self.soft_assign(test_px1)
                        test_q2 = self.soft_assign(test_px2)
                        lossY_test = self.pairwise_loss(test_q1, test_q2, test_type_values).item()
                        list_log[2] = lossY_test
                    self.train()

            total_pairwise_loss = 0.0
            temp_indices = np.random.permutation(np.arange(mergedCons.shape[0]))
            temp_indices = torch.from_numpy(temp_indices).to(self.device)
            reverse_temp_indices = torch.argsort(temp_indices.clone().detach())
            mergedCons = mergedCons[temp_indices]

            for cons_batch_idx in range(mergedCons_num_batch):
                start = cons_batch_idx * batch_size
                end = min(mergedCons_num, (cons_batch_idx + 1) * batch_size)
                batch = mergedCons[start:end]
                px1 = X[batch[:, 0]]
                px2 = X[batch[:, 1]]
                type_values = batch[:, 2]
                q1 = self.soft_assign(px1)
                q2 = self.soft_assign(px2)
                lossY = self.pairwise_loss(q1, q2, type_values)
                total_pairwise_loss += lossY * len(px1)
                optimizer.zero_grad()
                lossY.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
            total_pairwise_loss = total_pairwise_loss / mergedCons_num
            print("Pairwise Total Loss:", float(total_pairwise_loss.cpu()))

            mergedCons = mergedCons[reverse_temp_indices]

            # record log
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
        
        # save features
        if record_feature_dir is not None:
            features = self.get_features(X)
            torch.save(features, record_feature_dir)

        return final_epoch, final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari