import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cka import *
from model.hypersphere_loss import *

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# top 1 hard routing
def topk(t, k):
    values, index = t.topk(k=k, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

class Router(nn.Module):
    def __init__(self, input_dim, out_dim, device, strategy='topk'):
        super(Router, self).__init__()

        self.cls = nn.Linear(input_dim, out_dim, bias=False)
        self.out_dim = out_dim
        self.strategy = strategy
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        self.cls.weight = torch.nn.Parameter(self.cls.weight * 0)

    def forward(self, x):

        x = self.cls(x)
        # x = torch.sum(x, dim=1)
        if self.strategy == 'topk':
            self.noise = torch.rand(x.shape[0], self.out_dim).to(self.device)

        output = x + self.noise
        return output

class first_pre(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(first_pre, self).__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, feature_dim)
        )

    def forward(self, x):
        return self.pre(x)

class MoE(nn.Module):

    def __init__(self, n_label, in_dim, cluster_dim, expert_num, k, device, strategy='topk', is_train=False):
        super(MoE, self).__init__()
        self.router = Router(in_dim, expert_num, device, strategy=strategy)
        self.models = nn.ModuleList()
        self.k = k
        for i in range(expert_num):
            self.models.append(Encoder(in_dim, cluster_dim))
        self.strategy = strategy
        self.expert_num = expert_num
        self.device = device
        self.sim_threshold=1
        self.project = nn.Linear(cluster_dim, expert_num)
        self.clr_loss = CKALoss(cka_mode='kernel', sigma=0.8, device=self.device)

    def forward(self, x, t=True):
        gate_logits = self.router(x)
        select = F.softmax(gate_logits, dim=1)

        if self.strategy == 'topk':
            gate, index = topk(select, self.k)

        gate = gate/ gate.sum(dim=1, keepdim=True)
        mask = F.one_hot(index, self.expert_num).float()

        mask_1 = mask.sum(dim=1)

        density = mask_1.mean(dim=0)
        density_proxy = select.mean(dim=-2)
        balance_loss = (density_proxy * density).mean() * float(self.expert_num ** 2)

        mask_count = mask_1.sum(dim=-2, keepdim=True)
        mask_flat = mask.sum(dim=-1)

        combine_tensor = (gate[..., None, None] * mask_flat[..., None, None]
                          * F.one_hot(index, self.expert_num)[..., None])

        combine_tensor = combine_tensor.sum(dim=1)
        final_weight = combine_tensor.squeeze(-1)

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)
        expert_inputs = torch.einsum('sm,sec->esm', x, dispatch_tensor)

        output = []
        for i in range(self.expert_num):
            expert_output = self.models[i](expert_inputs[i])
            output.append(expert_output)

        output = torch.stack(output)
        expert_out = output.permute(1, 0, 2)
        selected_output = torch.einsum('bke,bed->bkd', mask, expert_out)

        dist_loss = 0
        if (t):
            all_sample_idx = [i for i in range(expert_out.shape[0])]
            sample_num = int(np.percentile(all_sample_idx, 50))
            random_sample_idx = random.sample(list(all_sample_idx), sample_num)
            sampled = selected_output[random_sample_idx]
            for i in range(self.k):
                for j in range(i+1, self.k):
                    out1 = sampled[:, i, :]
                    out1 = self.project(out1)
                    out2 = sampled[:, j, :]
                    out2 = self.project(out2)

                    cka_loss = self.clr_loss(out1, out2)
                    if self.sim_threshold < cka_loss:
                        print("Collapse Happen !!!!")
                        dist_loss += cka_loss

        final_output = torch.einsum('bn,bnd->bd', final_weight, expert_out)

        return final_output, select0, balance_loss, dist_loss



class MV_MOE(nn.Module):
    def __init__(self, num_view, input_dim_set, feature_dim, n_label, cluster_dim, expert_num, k, device, args, alpha=1.0, gamma=.1):
        super(MV_MOE, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.d = args.d
        self.view_num=num_view
        self.cluster_emb =cluster_emb = self.d * n_label
        self.s = None
        self.eta = args.eta
        self.n_clusters=n_label
        self.aggregation= aggregation = 'concat'
        self.is_train = False
        self.loss_fn=torch.nn.MSELoss()
        #self.align_loss = HypersphereLoss(t=2)
        self.select_align_loss = nn.CosineEmbeddingLoss()

        self.mv_pre=torch.nn.ModuleList()
        for i in range(num_view):
            self.mv_pre.append(nn.Sequential(first_pre(input_dim_set[i], feature_dim)))

        self.moe = MoE(n_label, feature_dim, cluster_dim, expert_num, k, device)
        self.mv_decoder=torch.nn.ModuleList()
        for i in range(num_view):
            decoder = nn.Sequential(
                nn.Linear(cluster_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, input_dim_set[i])
            )
            self.mv_decoder.append(decoder)

    def forward(self, x_set, t=True):
        middle_set=[]
        num=0
        for x in x_set:
            middle_set.append(self.mv_pre[num](x))
            num+=1
        sum_balance_loss = 0.0
        sum_dist_loss = 0.0
        sum_align_loss = 0.0
        gate_loss = 0.0
        view_h=[]
        reconstructed_x = []
        select_list = []

        for num in range(self.view_num):
            output, select, balance_loss, dist_loss = self.moe(middle_set[num], t)
            reconstructed_x.append(self.mv_decoder[num](output))
            select_list.append(select)
            view_h.append(output)
            sum_balance_loss+=balance_loss
            sum_dist_loss += dist_loss

        sum_align_loss = 0
        if(t):
            for u in range(self.view_num):
                for v in range(u, self.view_num):
                # Calculate cross-view native contrastive loss between view `u` and view `v` and accumulate it
                    sum_align_loss += self.align_loss(view_h[u], view_h[v])

        # sum_select_loss = 0
        # for i in range(self.view_num):
        #     for j in range(i + 1, self.view_num):
        #         # cosine embedding: label=+1 for positive pairs
        #         target = torch.ones(select_list[i].size(0), device=self.device)
        #         sum_select_loss += self.select_align_loss(
        #             select_list[i], select_list[j], target
        #         )

        if self.aggregation == 'concat':
            fused_h = torch.concat(view_h, dim=1)
            
        fused_h = fused_h.to(self.device)
        sum_balance_loss+=sum_align_loss

        return fused_h, reconstructed_x, sum_balance_loss, sum_dist_loss

    def get_results(self, data_loader, num_views):
        if num_views == 2:
            fused_h = []
            all_s = []
            all_y = []
            for idx, (feature_1, feature_2, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2]
                tmp_h, _, _, _= self.forward(feature_list, False)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 3:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3]
                tmp_h, _, _, _= self.forward(feature_list, False)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 4:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4]
                tmp_h, _, _, _= self.forward(feature_list, False)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 5:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, feature_5,feature_6, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5]
                tmp_h, _, _, _= self.forward(feature_list, False)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 6:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
                tmp_h, _, _, _= self.forward(feature_list, False)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        return fused_h, all_y

            

