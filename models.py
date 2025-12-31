import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import copy
import math 

class CGCDR(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, data_info,
                 src_num_clusters, tgt_num_clusters,
                 reg_weight: float = 1e-5,
                 alpha: float = 0.1, beta: float = 0.001):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_num_clusters = src_num_clusters
        self.tgt_num_clusters = tgt_num_clusters

        self.reg_weight = reg_weight
        self.alpha = alpha
        self.cl_weight = beta

        self.src_user_emb = nn.Embedding(num_users, emb_dim)
        self.src_item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.tgt_user_emb = nn.Embedding(num_users, emb_dim)
        self.tgt_item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)


        self.src_clusters = nn.Parameter(torch.randn(src_num_clusters, emb_dim))
        self.tgt_clusters = nn.Parameter(torch.randn(tgt_num_clusters, emb_dim))
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

        self.src2tgt_generator = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim, bias=False)
        )
        self.src2tgt_generator3 = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 3, bias=False),
            nn.GELU(),
            nn.Linear(emb_dim * 3, emb_dim, bias=False)
        )
        self.src2tgt_generator1 = nn.Sequential(
            nn.Linear(emb_dim * 1, emb_dim * 1, bias=False),
            nn.GELU(),
            nn.Linear(emb_dim * 1, emb_dim, bias=False)
        )

        self.fuse_score = nn.Linear(emb_dim, 1, bias=False)
        self.k_number = 0

        self.overlapped_num_users = data_info['overlapped_num_users']
        self.source_num_users = data_info['source_num_users']
        self.source_num_items = data_info['source_num_items']

        with torch.no_grad():
            self.src_user_emb.weight[self.source_num_users:].fill_(0)
            self.src_item_emb.weight[self.source_num_items+1:].fill_(0)
            self.tgt_user_emb.weight[self.overlapped_num_users: self.source_num_users].fill_(0)
            self.tgt_item_emb.weight[: self.source_num_items].fill_(0)

        self.bpr_loss = lambda pos, neg: -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

        nn.init.xavier_normal_(self.src_user_emb.weight)
        nn.init.xavier_normal_(self.src_item_emb.weight)
        nn.init.xavier_normal_(self.tgt_user_emb.weight)
        nn.init.xavier_normal_(self.tgt_item_emb.weight)
        nn.init.xavier_normal_(self.src_clusters)
        nn.init.xavier_normal_(self.tgt_clusters)

    
    @staticmethod
    def _pairwise_squared_distance(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        c: [K, D]
        return: [B, K]
        """
        xx = (x * x).sum(-1).unsqueeze(1).repeat(1, c.size(0))
        cc = (c * c).sum(-1).unsqueeze(0).repeat(x.size(0), 1)
        xc = x @ c.t()
        return xx + cc - 2 * xc

    def reg_loss(self, *embs):
        reg = 0
        for e in embs:
            reg += torch.norm(e, p=2) ** 2
        return reg

    def _cluster_loss(self, feats: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        L_cluster = mean(dist(feat, center))  -  mean_offdiag(dist(center, center))
        """
        feats = F.normalize(feats, p=2, dim=1)
        centers.data = F.normalize(centers.data, p=2, dim=1)

        sample_center_distance = self._pairwise_squared_distance(feats, centers)   # [B, K]
        sample_distance_loss = sample_center_distance.mean()

        center_center_distance = self._pairwise_squared_distance(centers, centers) # [K, K]
        K = centers.size(0)
        offdiag_mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
        center_offdiag_mean = center_center_distance[offdiag_mask].mean()
        center_distance_loss = -center_offdiag_mean

        return self.alpha * (sample_distance_loss + center_distance_loss)

    def _step_domain(self, uid, pos_ids, neg_ids, domain: str, train: bool):
        if domain == 'src':
            u_base = self.src_user_emb(uid)
            pos_emb = self.src_item_emb(pos_ids)
            neg_emb = self.src_item_emb(neg_ids)
            centers = self.src_clusters
        else:
            u_base = self.tgt_user_emb(uid)
            pos_emb = self.tgt_item_emb(pos_ids)
            neg_emb = self.tgt_item_emb(neg_ids)
            centers = self.tgt_clusters

        pos_score = torch.sum(u_base * pos_emb, dim=1)
        neg_score = torch.sum(u_base * neg_emb, dim=1)
        loss = self.bpr_loss(pos_score, neg_score)
        if train:
            loss += self._cluster_loss(u_base, centers) + self.reg_weight * self.reg_loss(u_base, pos_emb, neg_emb)
            # loss += self.reg_weight * self.reg_loss(u_base, pos_emb, neg_emb)
        return loss

    def forward(self, uid, src_ids, pos_ids, neg_ids, stage=None):
        if stage in ['train_src', 'val_src']:
            return self._step_domain(uid, pos_ids, neg_ids, domain='src', train=('train' in stage))
        elif stage in ['train_tgt', 'val_tgt']:
            return self._step_domain(uid, pos_ids, neg_ids, domain='tgt', train=('train' in stage))
        
        elif stage in ['overlap', 'eval_overlap']:

            uids = uid
            # src_ids: [B, L]
            item_seq = src_ids  # [B, L]

            u_src = self.src_user_emb(uids)  # [B, D]
            seq_emb = self.src_item_emb(item_seq)  # [B, L, D]

            k = self.k_number  
            seq_emb = seq_emb[:, :k, :]  #  [B, k, D]

            B, L, D = seq_emb.shape
            # seq_flat = seq_emb.view(B * L, D)
            seq_flat = seq_emb.reshape(B * L, D)

            centers_src = self.src_clusters  # [K_s, D]
            Ks = centers_src.size(0)
            
            # pairwise distances between each item and each cluster center
            seq_flat = seq_emb.reshape(B * L, D) # [B*L, D]
            dist = self._pairwise_squared_distance(seq_flat, centers_src) # [B*L, Ks]
            dist = dist.view(B, L, Ks)

            # print(f'dist: {dist[:10]}')
            # convert to similarity and soft-assign each item to clusters
            sim = -dist 
            tau = 0.1
            w = F.softmax(sim / (tau + 1e-12), dim=-1) # [B, L, Ks]
            # weighted aggregation to get cluster-level summaries per user
            # summaries: [B, Ks, D]
            w_t = w.permute(0, 2, 1) # [B, Ks, L]
            summaries = torch.matmul(w_t, seq_emb) # [B, Ks, D]
            norm = w_t.sum(dim=-1, keepdim=True).clamp(min=1e-6) # [B, Ks, 1]
            summaries = summaries / norm # [B, Ks, D]
            # compute cluster activity (how much each cluster is supported by user's items)
            cluster_activity = norm.squeeze(-1) # [B, Ks]
            top_M = 10
            M = min(top_M, Ks)
            topk_vals, topk_idx = torch.topk(cluster_activity, M, dim=1) # [B, M]

            # gather top-M summaries and top-M centers
            batch_idx = torch.arange(B).unsqueeze(1)
            topk_summaries = summaries[batch_idx, topk_idx] # [B, M, D]
            topk_centers = centers_src[topk_idx.view(-1)].view(B, M, D)
            # generate candidates per selected cluster
            # add small gaussian noise to summaries to inject diversity
            z = torch.randn_like(topk_summaries) * 0.05

            u_src_expanded = u_src.unsqueeze(1).expand(-1, M, -1)  # [B, M, D]
            gen_inp = torch.cat([topk_summaries + z, topk_centers, u_src_expanded], dim=-1) # [B, M, 3D]
            gen_inp = gen_inp.view(B * M, 3 * D)

            # Ablation--woCG、woUsrc
            # with
            gen_out = self.src2tgt_generator3(gen_inp) # [B*M, D]
            # woCG
            # gen_out = self.src2tgt_generator1(u_src_expanded) # [B*M, D]   
            # woUsrc
            # gen_out = self.src2tgt_generator(torch.cat([topk_summaries + z, topk_centers], dim=-1).view(B * M, 2 * D)) # [B*M, D]
            # woCC
            # gen_out = self.src2tgt_generator(torch.cat([topk_summaries + z, u_src_expanded], dim=-1).view(B * M, 2 * D)) # [B*M, D]

            gen_out = gen_out.view(B, M, D) # [B, M, D]

            # scoring (attention) over the M candidates per user
            # fuse_score is a linear layer: [D] -> 1
            score_in = gen_out
            score = self.fuse_score(score_in).squeeze(-1) # [B, M]
            
            mask_valid = (topk_vals > 1e-6).float() # [B, M]
            # set invalid scores to -inf so softmax ignores them
            score = score.masked_fill(mask_valid == 0, -1e9)

            attn = F.softmax(score, dim=1) # [B, M]
            u_pred = torch.bmm(attn.unsqueeze(1), gen_out).squeeze(1) # [B, D]

            if stage == 'eval_overlap':
                return u_pred

            u_tgt_emd = self.tgt_user_emb(uids)  # [B, D]
            loss = F.mse_loss(u_pred, u_tgt_emd)
            # return loss 
            # ========= Cluster-Assisted Contrastive Loss =========
            # embeddings
            pos_emb = self.tgt_item_emb(pos_ids)  # [B, D]
            neg_emb = self.tgt_item_emb(neg_ids)  # [B, D]

            dist = self._pairwise_squared_distance(neg_emb, self.tgt_clusters)  # [B, K_t]
            neg_cluster_idx = torch.argmin(dist, dim=1)
            neg_cluster_emb = self.tgt_clusters[neg_cluster_idx]  # [B, D]

            pos_score = torch.sum(u_pred * pos_emb, dim=1, keepdim=True)          # [B,1]
            neg_score = torch.sum(u_pred * neg_cluster_emb, dim=1, keepdim=True)  # [B,1]

            tau = 0.1
            logits = torch.cat([pos_score, neg_score], dim=1) / tau  # [B,2]
            labels = torch.zeros(u_pred.size(0), dtype=torch.long, device=u_pred.device)  
            cl_loss = F.cross_entropy(logits, labels)

            loss = loss + self.cl_weight * cl_loss

            return loss 
        else:
            raise ValueError(f'Unknown stage {stage}')








    






