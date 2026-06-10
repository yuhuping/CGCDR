import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disco_utils import (
    DynamicInteractionDataset,
    InteractionDataset,
    OverlapUserDataset,
)
from utils import TestSeqItemDataset, sample_candidates


class DisCoTrainer:
    def __init__(self, model, args, data_info):
        self.model = model
        self.args = args
        self.data_info = data_info
        self.data_root = os.path.join("data", args.Task)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_start = data_info["source_num_items"] + 1
        self.target_end = data_info["total_num_items"]
        self.output_path = os.path.join("saved", args.Task)
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_suffix = f"_{args.info}" if args.info else ""
        self.checkpoint_path = os.path.join(
            self.output_path, f"DisCo{checkpoint_suffix}_best.pt"
        )
        self.logger = logging.getLogger(f"{args.Task}.DisCoTrainer")

    @staticmethod
    def _cycle(loader):
        while True:
            yield from loader

    def _move_batch(self, batch):
        return tuple(value.to(self.device) for value in batch)

    @torch.no_grad()
    def evaluate_leave_one_out(
        self, dataset, negative_samples=999, description="DisCo leave-one-out"
    ):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()
        hits = {1: 0, 5: 0, 10: 0}
        ndcgs = {1: 0.0, 5: 0.0, 10: 0.0}

        for index, (users, _, positive, _, target_history) in enumerate(
            tqdm(loader, desc=description, ncols=100)
        ):
            candidates = sample_candidates(
                pos_id=positive.numpy(),
                pos_ids=target_history.numpy().flatten().tolist(),
                MIN=self.target_start,
                MAX=self.target_end,
                neg_sample_size=negative_samples,
                seed=self.args.eval_seed + index,
            )
            users = users.to(self.device)
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            scores = self.model.cross_domain_logits(
                users, candidate_tensor
            ).squeeze(0)
            ranking = torch.argsort(scores, descending=True)
            rank = (ranking == 0).nonzero(as_tuple=True)[0].item()
            for k in hits:
                if rank < k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / np.log2(rank + 2)

        total = len(dataset)
        metrics = {}
        for k in hits:
            metrics[f"HR@{k}"] = hits[k] / total
            metrics[f"NDCG@{k}"] = ndcgs[k] / total
        return metrics

    def main(self):
        self.model.to(self.device)
        source_path = os.path.join(self.data_root, "stage1_train_src.csv")
        target_path = os.path.join(self.data_root, "stage1_train_tgt.csv")
        if self.args.dynamic_neg_sampling:
            source_data = DynamicInteractionDataset(
                source_path,
                minimum_item=1,
                maximum_item=self.data_info["source_num_items"],
                seed=self.args.seed,
            )
            target_data = DynamicInteractionDataset(
                target_path,
                minimum_item=self.target_start,
                maximum_item=self.target_end,
                seed=self.args.seed + 1,
            )
        else:
            source_data = InteractionDataset(source_path)
            target_data = InteractionDataset(target_path)
        overlap_data = OverlapUserDataset(
            source_path,
            target_path,
            self.data_info["overlapped_num_users"],
        )
        validation_data = TestSeqItemDataset(
            os.path.join(self.data_root, "stage1_val.csv")
        )
        test_data = TestSeqItemDataset(
            os.path.join(self.data_root, "stage1_test.csv")
        )

        loader_args = dict(
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        source_loader = DataLoader(source_data, shuffle=True, **loader_args)
        target_loader = DataLoader(target_data, shuffle=True, **loader_args)
        overlap_loader = DataLoader(
            overlap_data,
            shuffle=True,
            batch_size=min(self.args.batch_size, len(overlap_data)),
            drop_last=len(overlap_data) >= self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        best_hr = -1.0
        best_ndcg = -1.0
        stale_epochs = 0

        for epoch in range(self.args.epoch):
            if self.args.dynamic_neg_sampling:
                source_data.set_epoch(epoch)
                target_data.set_epoch(epoch)
            self.model.train()
            source_iterator = self._cycle(source_loader)
            target_iterator = self._cycle(target_loader)
            overlap_iterator = self._cycle(overlap_loader)
            steps = max(len(source_loader), len(target_loader))
            totals = dict(loss=0.0, source_rec=0.0, target_rec=0.0, contrast=0.0)

            progress = tqdm(
                range(steps), desc=f"DisCo Epoch {epoch}", ncols=100
            )
            for _ in progress:
                source_batch = self._move_batch(next(source_iterator))
                target_batch = self._move_batch(next(target_iterator))
                overlap_users = next(overlap_iterator).to(self.device)

                loss, parts = self.model.total_loss(
                    source_batch, target_batch, overlap_users
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_clip
                )
                optimizer.step()
                self.model.update_momentum_encoders()

                totals["loss"] += loss.item()
                for name in parts:
                    totals[name] += parts[name].item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            averages = {name: value / steps for name, value in totals.items()}
            validation = self.evaluate_leave_one_out(
                validation_data,
                negative_samples=self.args.eval_negatives,
                description=f"DisCo validation {epoch}",
            )
            self.logger.info(
                "Epoch %d loss %.4f src %.4f tgt %.4f contrast %.4f "
                "val HR@10 %.4f NDCG@10 %.4f",
                epoch,
                averages["loss"],
                averages["source_rec"],
                averages["target_rec"],
                averages["contrast"],
                validation["HR@10"],
                validation["NDCG@10"],
            )

            improved = validation["HR@10"] > best_hr
            improved = improved or (
                validation["HR@10"] == best_hr
                and validation["NDCG@10"] > best_ndcg
            )
            if improved:
                best_hr = validation["HR@10"]
                best_ndcg = validation["NDCG@10"]
                stale_epochs = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self.args.stopping_step:
                    break

        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError("DisCo training did not produce a checkpoint")
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        metrics = self.evaluate_leave_one_out(
            test_data, negative_samples=self.args.eval_negatives
        )
        for k in (1, 5, 10):
            self.logger.info(
                "HR@%d: %.4f, NDCG@%d: %.4f",
                k,
                metrics[f"HR@{k}"],
                k,
                metrics[f"NDCG@{k}"],
            )
