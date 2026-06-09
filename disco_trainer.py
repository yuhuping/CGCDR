import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disco_utils import InteractionDataset, OverlapUserDataset
from utils import TestSeqItemDataset, SeqItemDataset, sample_candidates


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
        self.checkpoint_path = os.path.join(self.output_path, "DisCo_best.pt")
        self.logger = logging.getLogger(f"{args.Task}.DisCoTrainer")

    @staticmethod
    def _cycle(loader):
        while True:
            yield from loader

    def _move_batch(self, batch):
        return tuple(value.to(self.device) for value in batch)

    def _validation_loss(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for users, _, positive, negative in loader:
                users = users.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                loss = self.model.cross_domain_pair_loss(
                    users, positive, negative
                )
                total_loss += loss.item()
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def evaluate_leave_one_out(self, dataset, negative_samples=999):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()
        hits = {1: 0, 5: 0, 10: 0}
        ndcgs = {1: 0.0, 5: 0.0, 10: 0.0}

        for users, _, positive, _, target_history in tqdm(
            loader, desc="DisCo leave-one-out", ncols=100
        ):
            candidates = sample_candidates(
                pos_id=positive.numpy(),
                pos_ids=target_history.numpy().flatten().tolist(),
                MIN=self.target_start,
                MAX=self.target_end,
                neg_sample_size=negative_samples,
                seed=self.args.seed + 2,
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
            self.logger.info(
                "HR@%d: %.4f, NDCG@%d: %.4f",
                k,
                metrics[f"HR@{k}"],
                k,
                metrics[f"NDCG@{k}"],
            )
        return metrics

    def main(self):
        self.model.to(self.device)
        source_data = InteractionDataset(
            os.path.join(self.data_root, "stage1_train_src.csv")
        )
        target_data = InteractionDataset(
            os.path.join(self.data_root, "stage1_train_tgt.csv")
        )
        overlap_data = OverlapUserDataset(
            os.path.join(self.data_root, "stage1_train_meta.csv")
        )
        validation_data = SeqItemDataset(
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
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        best_validation = float("inf")
        stale_epochs = 0

        for epoch in range(self.args.epoch):
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
            validation_loss = self._validation_loss(validation_loader)
            self.logger.info(
                "Epoch %d loss %.4f src %.4f tgt %.4f contrast %.4f val %.4f",
                epoch,
                averages["loss"],
                averages["source_rec"],
                averages["target_rec"],
                averages["contrast"],
                validation_loss,
            )

            if validation_loss < best_validation:
                best_validation = validation_loss
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
        self.evaluate_leave_one_out(
            test_data, negative_samples=self.args.eval_negatives
        )
