import csv
import os
import tempfile
import unittest

import torch

from disco import DisCo
from disco_utils import (
    DynamicInteractionDataset,
    InteractionDataset,
    OverlapUserDataset,
    build_domain_graph,
)


class DisCoTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_info = {
            "overlapped_num_users": 3,
            "source_num_users": 5,
            "total_num_users": 7,
            "source_num_items": 4,
            "total_num_items": 8,
        }
        self.source_path = os.path.join(self.temp_dir.name, "source.csv")
        self.target_path = os.path.join(self.temp_dir.name, "target.csv")
        self._write(
            self.source_path,
            [(0, 1, 2), (0, 2, 3), (1, 2, 4), (2, 3, 1), (3, 4, 1)],
        )
        self._write(
            self.target_path,
            [(0, 5, 6), (0, 6, 7), (1, 6, 8), (2, 7, 5), (5, 8, 5)],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write(path, rows):
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["uid", "pos_iid", "neg_iid"])
            writer.writerows(rows)

    def test_loss_backward_and_cross_domain_scoring(self):
        source_graph = build_domain_graph(
            self.source_path, self.data_info, "src", max_neighbors=2
        )
        target_graph = build_domain_graph(
            self.target_path, self.data_info, "tgt", max_neighbors=2
        )
        model = DisCo(
            num_users=7,
            source_graph=source_graph,
            target_graph=target_graph,
            embedding_dim=8,
            num_intents=2,
            random_walk_steps=2,
            dropout=0.0,
        )
        users = torch.tensor([0, 1])
        source_batch = (
            users,
            torch.tensor([1, 2]),
            torch.tensor([4, 3]),
        )
        target_batch = (
            users,
            torch.tensor([5, 6]),
            torch.tensor([8, 7]),
        )

        loss, parts = model.total_loss(source_batch, target_batch, users)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"source_rec", "target_rec", "contrast"})
        loss.backward()
        model.update_momentum_encoders()

        model.eval()
        scores = model.cross_domain_logits(
            users, torch.tensor([[5, 6], [7, 8]])
        )
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fixed_negatives_and_overlap_users(self):
        dataset = InteractionDataset(self.source_path)
        self.assertEqual(int(dataset[0][2]), 2)
        overlap = OverlapUserDataset(
            self.source_path, self.target_path, overlapped_num_users=3
        )
        self.assertEqual(set(overlap.users.tolist()), {0, 1, 2})

    def test_dynamic_negatives_are_valid_and_reproducible(self):
        first = DynamicInteractionDataset(self.source_path, 1, 4, seed=2025)
        second = DynamicInteractionDataset(self.source_path, 1, 4, seed=2025)
        epoch_zero = [int(first[index][2]) for index in range(len(first))]
        self.assertEqual(
            epoch_zero,
            [int(second[index][2]) for index in range(len(second))],
        )

        positives = {(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)}
        for index, negative in enumerate(epoch_zero):
            self.assertNotIn((int(first.users[index]), negative), positives)

        first.set_epoch(1)
        epoch_one = [int(first[index][2]) for index in range(len(first))]
        self.assertNotEqual(epoch_zero, epoch_one)


if __name__ == "__main__":
    unittest.main()
