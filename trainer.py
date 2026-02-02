import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
import sys
import os
import logging

from utils import *
from models import *

import gc
class CGCDRTrainer():
    def __init__(self, model, args, data_info):

        self.model = model
        self.model_name = 'CGCDR'
        self.data_root = './data/' + args.Task + '/'
        self.epoch = args.epoch
        self.lr = args.lr
        self.stopping_step = args.stopping_step
        self.val_ratio = args.val_ratio
        self.seed = args.seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_start = data_info['source_num_items'] + 1
        self.target_end = data_info['total_num_items']
        self.out_path = os.path.join(os.getcwd(), f"saved/{args.Task}")
        os.makedirs(self.out_path, exist_ok=True)
        self.logger = logging.getLogger(f"{args.Task}.CGCDRTrainer")

    @torch.no_grad()
    def eval_leave_one_out(self, data, neg_sample_size=999, hit_ks=(1, 5, 10)):

        loader = DataLoader(data, batch_size=1, shuffle=False)
        self.model.eval()

        hits = {k: 0 for k in hit_ks}
        ndcgs = {k: 0.0 for k in hit_ks}
        total = 0

        for uid, src_ids, pos_id, _, tgt_iids in tqdm(loader, desc="leave-one-out Eval"):
            uid = uid.to(self.device)
            src_ids = src_ids.to(self.device)
            pos_id = pos_id.to(self.device)
            tgt_iids = tgt_iids.to(self.device)

            u_emb = self.model(uid, src_ids, pos_id, neg_ids=None, stage='eval_overlap').squeeze(0)  # [D]

            pos_id_np = pos_id.cpu().numpy()
            tgt_iids_np = tgt_iids.cpu().numpy().flatten()

            cand_ids = sample_candidates(
                pos_id=pos_id_np,
                pos_ids=tgt_iids_np.tolist(),
                MIN=self.target_start,
                MAX=self.target_end,
                neg_sample_size=neg_sample_size,
                seed=self.seed + 2
            )

            candidate_tensor = torch.tensor(cand_ids, dtype=torch.long, device=self.device)
            candidate_embs = self.model.tgt_item_emb(candidate_tensor)  # [1000, D]
            scores = torch.matmul(u_emb, candidate_embs.t()).squeeze()  # [1000]

            _, sorted_indices = torch.sort(scores, descending=True)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()

            total += 1
            for k in hit_ks:
                if rank < k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / np.log2(rank + 2)
                else:
                    ndcgs[k] += 0.0

        self.logger.info("\n=== Leave-One-Out Evaluation Results ===")
        self.logger.info(f"Evaluated {total} samples with {neg_sample_size} negative samples each")
        for k in hit_ks:
            hr = hits[k] / total
            ndcg = ndcgs[k] / total
            self.logger.info(f"HR@{k}: {hr:.4f} ({hits[k]}/{total})")
            self.logger.info(f"NDCG@{k}: {ndcg:.4f}")

        return {f"HR@{k}": hits[k] / total for k in hit_ks}
    
    def _run_kmeans(self, feats: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
        """
        feats: [N, D] on device
        return: centers [K, D] on device
        """
        N = feats.size(0)
        idx = torch.randperm(N, device=feats.device)[:k]
        centers = feats[idx].clone()  # [K, D]
        for _ in range(iters):
            # [N, K] squared distances
            xx = (feats * feats).sum(-1, keepdim=True)         # [N, 1]
            cc = (centers * centers).sum(-1).unsqueeze(0)      # [1, K]
            xc = feats @ centers.t()                            # [N, K]
            dist = xx + cc - 2 * xc
            assign = torch.argmin(dist, dim=1)                  # [N]
            # recompute centers
            new_centers = []
            for j in range(k):
                mask = (assign == j)
                if mask.any():
                    new_centers.append(feats[mask].mean(dim=0, keepdim=True))
                else:
                    # empty cluster -> random re-init
                    ridx = torch.randint(0, N, (1,), device=feats.device)
                    new_centers.append(feats[ridx])
            centers = torch.cat(new_centers, dim=0)
        return centers

    @torch.no_grad()
    def _prepare_kmeans_for_domain(self, loader, domain: str, k: int):
        self.model.eval()
        feats = []
        for uids, *_ in tqdm(loader, desc=f"KMeans init ({domain})", ncols=100):
            uids = uids.to(self.device)
            if domain == 'src':
                u = self.model.src_user_emb(uids)
            else:
                u = self.model.tgt_user_emb(uids)
            feats.append(u)
        if len(feats) == 0:
            return
        feats = torch.cat(feats, dim=0)  # [N, D] on device
        centers = self._run_kmeans(feats, k=k, iters=20)
        if domain == 'src':
            self.model.src_clusters.data.copy_(centers)
        else:
            self.model.tgt_clusters.data.copy_(centers)

    def main(self):
        self.model = self.model.to(self.device)
        def split_ds(ds):
            total = len(ds)
            val_size = int(total * self.val_ratio)
            train_size = total - val_size
            return random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
        if self.epoch != 0:

            src_full = SeqItemDataset(os.path.join(self.data_root, 'stage1_train_src.csv'))
            tgt_full = SeqItemDataset(os.path.join(self.data_root, 'stage1_train_tgt.csv'))

            k_src = self.model.src_clusters.size(0)
            k_tgt = self.model.tgt_clusters.size(0)

            full_src_loader = DataLoader(src_full, batch_size=2048, shuffle=False)
            full_tgt_loader = DataLoader(tgt_full, batch_size=2048, shuffle=False)
            # self._prepare_kmeans_for_domain(full_src_loader, 'src', k_src)
            # self._prepare_kmeans_for_domain(full_tgt_loader, 'tgt', k_tgt)
            gc.collect()
            torch.cuda.empty_cache()
            
            train_src, val_src = split_ds(src_full)
            train_tgt, val_tgt = split_ds(tgt_full)
            train_src_loader = DataLoader(train_src, batch_size=2048, shuffle=True)
            val_src_loader = DataLoader(val_src, batch_size=2048, shuffle=False)
            train_tgt_loader = DataLoader(train_tgt, batch_size=2048, shuffle=True)
            val_tgt_loader = DataLoader(val_tgt, batch_size=2048, shuffle=False)

        meta_full = SeqItemDataset(os.path.join(self.data_root, 'stage1_train_meta.csv'))
        
        data_val  = SeqItemDataset(os.path.join(self.data_root, 'stage1_val.csv'))
        data_test = TestSeqItemDataset(os.path.join(self.data_root, 'stage1_test.csv'))
        # train_meta, val_meta = split_ds(meta_full)
        train_meta_loader = DataLoader(meta_full, batch_size=1024, shuffle=True)
        val_meta_loader = DataLoader(data_val, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_metric = float('inf')
        count = 0
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0.0
            for uids, src_ids, pos_ids, neg_ids in tqdm(train_src_loader, desc=f"SRC Epoch {epoch}", ncols=100):
                uids, pos_ids, neg_ids = uids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                loss = self.model(uids, None, pos_ids, neg_ids, stage='train_src')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.logger.info(f"SRC Epoch {epoch} Train Loss: {total_loss/len(train_src_loader):.4f}")

            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for uids, src_ids, pos_ids, neg_ids in val_src_loader:
                    uids, pos_ids, neg_ids = uids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                    loss = self.model(uids, None, pos_ids, neg_ids, stage='val_src')
                    eval_loss += loss.item()
            eval_loss /= max(1, len(val_src_loader))
            self.logger.info(f"SRC Val Loss: {eval_loss:.4f}")

            if eval_loss < best_metric:
                best_metric = eval_loss
                count = 0
                torch.save(self.model.state_dict(), os.path.join(self.out_path, f'{self.model_name}_pretrain.pt'))
            else:
                count += 1
                if count >= self.stopping_step:
                    break

        self.model.load_state_dict(torch.load(os.path.join(self.out_path, f'{self.model_name}_pretrain.pt')), strict=False)
        best_metric = float('inf')
        count = 0
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0.0
            for uids, src_ids, pos_ids, neg_ids in tqdm(train_tgt_loader, desc=f"TGT Epoch {epoch}", ncols=100):
                uids, pos_ids, neg_ids = uids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                loss = self.model(uids, None, pos_ids, neg_ids, stage='train_tgt')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.logger.info(f"TGT Epoch {epoch} Train Loss: {total_loss/len(train_tgt_loader):.4f}")

            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for uids, src_ids, pos_ids, neg_ids in val_tgt_loader:
                    uids, pos_ids, neg_ids = uids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                    loss = self.model(uids, None, pos_ids, neg_ids, stage='val_tgt')
                    eval_loss += loss.item()
            eval_loss /= max(1, len(val_tgt_loader))
            self.logger.info(f"TGT Val Loss: {eval_loss:.4f}")

            if eval_loss < best_metric:
                best_metric = eval_loss
                count = 0
                torch.save(self.model.state_dict(), os.path.join(self.out_path, f'{self.model_name}_pretrain.pt'))
            else:
                count += 1
                if count >= self.stopping_step:
                    break
        
        for k in [20]:  # for Parameter Analysis
            self.model.load_state_dict(torch.load(os.path.join(self.out_path, f'{self.model_name}_pretrain.pt')), strict=False)
            self.logger.info(f"Start with K_number={k}")
            self.model.k_number = k
            optimizer_overlap = torch.optim.Adam(
                list(self.model.src2tgt_generator.parameters()) + 
                list(self.model.src2tgt_generator3.parameters()) + 
                list(self.model.src2tgt_generator1.parameters()) +
                list(self.model.mapping.parameters()) +
                list(self.model.fuse_score.parameters()),
                lr=self.lr 
            )
            best_metric = float('inf')
            count = 0
            for epoch in range(1000):
                self.model.train()
                total_loss = 0.0
                for uids, src_ids, pos_ids, neg_ids in tqdm(train_meta_loader, desc=f"OVER Epoch {epoch}", ncols=100):
                    uids, src_ids, pos_ids, neg_ids = uids.to(self.device), src_ids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                    loss = self.model(uids, src_ids, pos_ids, neg_ids, stage='overlap')
                    optimizer_overlap.zero_grad()
                    loss.backward()
                    optimizer_overlap.step()
                    total_loss += loss.item()
                # self.logger.info(f"OVER Epoch {epoch} Train Loss: {total_loss/len(train_meta_loader):.4f}")

                self.model.eval()
                eval_loss = 0.0
                with torch.no_grad():
                    for uids, src_ids, pos_ids, neg_ids in val_meta_loader:
                        uids, src_ids, pos_ids, neg_ids = uids.to(self.device), src_ids.to(self.device), pos_ids.to(self.device), neg_ids.to(self.device)
                        loss = self.model(uids, src_ids, pos_ids, neg_ids, stage='overlap')
                        eval_loss += loss.item()
                eval_loss /= max(1, len(val_meta_loader))
                # self.logger.info(f"OVER Val Loss: {eval_loss:.4f}")

                if eval_loss < best_metric:
                    self.logger.info(f"best OVER Val Loss: {eval_loss:.4f}")
                    best_metric = eval_loss
                    count = 0
                    torch.save(self.model.state_dict(), os.path.join(self.out_path, f'{self.model_name}_best.pt'))

                else:
                    count += 1
                    if count >= self.stopping_step:
                        break

            self.model.load_state_dict(torch.load(os.path.join(self.out_path, f'{self.model_name}_best.pt')))
            self.eval_leave_one_out(data_test, neg_sample_size=999)
