import os
import random

import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.evaluate_embedding import cluster_acc

class MV_MOE_Trainer:

    def __init__(self, args, model, optimizer, num_view, lamda, gamma, device):
        """Initialize DMVC-CE Trainer

        Parameters
        ----------
        model: DMCV-CE model.
        optimizer: Optimizer of the proposed model.
        num_view: The number of total data views.
        device: torch.device object for device to use.
        lamda: Weight of equilibrium loss.
        gamma: Weight of the distinctiveness enhancer loss.
        args: The arguments of network.
        """
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.gamma = gamma
        self.device = device
        self.args = args
        self.num_view = num_view

    def train(self, dataloader, learning_rate, total_epochs):

        """
        Training function
        Parameters
        ----------
        dataloader: Dataloader object for the training dataset.
        learning_rate: Initial learning rate for training.
        total_epochs: Total epochs for training.
        """
        best_acc = -np.inf
        best_nmi = -np.inf
        best_ari = -np.inf
        align_const = 0.01
        log_interval = 1
        loss_fn=torch.nn.MSELoss()
        pbar = tqdm(range(1, total_epochs + 1))
        for epoch in pbar:
            loss_all = 0
            self.model.train()
            for idx, data in enumerate(dataloader):
                if self.num_view ==2:
                    feature_1, feature_2, y = data
                    feature_list = [feature_1, feature_2]
                elif self.num_view ==3:
                    feature_1, feature_2, feature_3, y = data
                    feature_list = [feature_1, feature_2, feature_3]
                elif self.num_view ==4:
                    feature_1, feature_2, feature_3, feature_4, y = data
                    feature_list = [feature_1, feature_2, feature_3, feature_4]
                elif self.num_view ==5:
                    feature_1, feature_2, feature_3, feature_4, feature_5, y = data
                    feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5]
                elif self.num_view ==6:
                    feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, y = data
                    feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
                self.optimizer.zero_grad()

                z, reconstructed_x, balance_loss, distinctive_loss, align_loss = self.model(feature_list, y)
                reconstruction_loss=0
                for i in range(self.num_view):
                    reconstruction_loss += loss_fn(data[i],reconstructed_x[i])

                loss = reconstruction_loss + self.lamda*balance_loss + self.gamma* distinctive_loss + align_const*align_loss
                #loss = reconstruction_loss
                # Total loss
                loss_all += loss.item()
                loss.backward()
                self.optimizer.step()
            if epoch % 20 == 0:
                self.model.eval()
                emb, y = self.model.get_results(dataloader, self.num_view)
                
                n_cluster = len(np.unique(y.numpy()))
                kmeans = KMeans(n_clusters=n_cluster, n_init=100)
                y_pred = kmeans.fit_predict(emb.cpu().detach().numpy())
                y=y.cpu().detach().numpy()
                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                if best_acc<acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    self.save('./BDGP.pth')
            if epoch % 20 == 0:
                pbar.set_description(
                    "Epoch {}| # Total Loss: {:.4}".format(
                        epoch,
                        loss_all
                    )
                )
        return best_acc, best_nmi, best_ari

    def save(self, path):
        torch.save({'model': self.model.state_dict()}, os.path.join(path))
