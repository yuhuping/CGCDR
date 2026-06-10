import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _sample_neighbors(keys, values, num_nodes, max_neighbors, rng):
    adjacency = np.full((num_nodes, max_neighbors), -1, dtype=np.int64)
    if len(keys) == 0:
        return adjacency

    permutation = rng.permutation(len(keys))
    shuffled_keys = keys[permutation]
    shuffled_values = values[permutation]
    order = np.argsort(shuffled_keys, kind="stable")
    sorted_keys = shuffled_keys[order]
    sorted_values = shuffled_values[order]

    positions = np.arange(len(sorted_keys))
    group_start = np.where(
        np.r_[True, sorted_keys[1:] != sorted_keys[:-1]], positions, 0
    )
    group_start = np.maximum.accumulate(group_start)
    rank = positions - group_start
    keep = rank < max_neighbors
    adjacency[sorted_keys[keep], rank[keep]] = sorted_values[keep]
    return adjacency


def build_domain_graph(csv_path, data_info, domain, max_neighbors=10, seed=2025):
    """Build capped CPU adjacency tables from the positive interaction graph."""
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    frame = pd.read_csv(csv_path, usecols=["uid", "pos_iid"])
    users = frame["uid"].to_numpy(dtype=np.int64, copy=False)
    global_items = frame["pos_iid"].to_numpy(dtype=np.int64, copy=False)

    if domain == "src":
        item_offset = 1
        num_items = data_info["source_num_items"]
    elif domain == "tgt":
        item_offset = data_info["source_num_items"] + 1
        num_items = data_info["total_num_items"] - data_info["source_num_items"]
    else:
        raise ValueError("domain must be 'src' or 'tgt'")

    items = global_items - item_offset
    valid = (
        (users >= 0)
        & (users < data_info["total_num_users"])
        & (items >= 0)
        & (items < num_items)
    )
    users = users[valid]
    items = items[valid]

    # Duplicate positive pairs do not add graph information.
    pairs = np.unique(np.stack([users, items], axis=1), axis=0)
    users = pairs[:, 0]
    items = pairs[:, 1]

    user_degree = np.bincount(
        users, minlength=data_info["total_num_users"]
    ).astype(np.float32)
    item_degree = np.bincount(items, minlength=num_items).astype(np.float32)
    rng = np.random.RandomState(seed)

    return {
        "user_neighbors": torch.from_numpy(
            _sample_neighbors(
                users,
                items,
                data_info["total_num_users"],
                max_neighbors,
                rng,
            )
        ),
        "item_neighbors": torch.from_numpy(
            _sample_neighbors(items, users, num_items, max_neighbors, rng)
        ),
        "user_degree": torch.from_numpy(np.maximum(user_degree, 1.0)),
        "item_degree": torch.from_numpy(np.maximum(item_degree, 1.0)),
        "item_offset": item_offset,
        "num_items": num_items,
    }


class InteractionDataset(Dataset):
    def __init__(self, csv_path):
        frame = pd.read_csv(csv_path, usecols=["uid", "pos_iid", "neg_iid"])
        self.users = torch.from_numpy(
            frame["uid"].to_numpy(dtype=np.int64, copy=True)
        )
        self.positive_items = torch.from_numpy(
            frame["pos_iid"].to_numpy(dtype=np.int64, copy=True)
        )
        self.negative_items = torch.from_numpy(
            frame["neg_iid"].to_numpy(dtype=np.int64, copy=True)
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return (
            self.users[index],
            self.positive_items[index],
            self.negative_items[index],
        )


class DynamicInteractionDataset(Dataset):
    """Deterministic epoch-wise uniform negative sampling."""

    def __init__(self, csv_path, minimum_item, maximum_item, seed):
        frame = pd.read_csv(csv_path, usecols=["uid", "pos_iid"])
        users = frame["uid"].to_numpy(dtype=np.int64, copy=True)
        positive_items = frame["pos_iid"].to_numpy(dtype=np.int64, copy=True)
        self.users = torch.from_numpy(users)
        self.positive_items = torch.from_numpy(positive_items)
        self.minimum_item = minimum_item
        self.item_count = maximum_item - minimum_item + 1
        self.seed = int(seed)
        self.epoch = 0

        local_items = positive_items - minimum_item
        self.positive_codes = np.unique(users * self.item_count + local_items)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.users)

    @staticmethod
    def _mix(value):
        value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)

    def _is_positive(self, code):
        position = np.searchsorted(self.positive_codes, code)
        return (
            position < len(self.positive_codes)
            and self.positive_codes[position] == code
        )

    def __getitem__(self, index):
        user = int(self.users[index])
        base = (
            self.seed
            + self.epoch * 0xD1B54A32D192ED03
            + int(index) * 0x94D049BB133111EB
        )
        attempt = 0
        while True:
            local_item = self._mix(base + attempt) % self.item_count
            code = user * self.item_count + local_item
            if not self._is_positive(code):
                break
            attempt += 1
        negative_item = self.minimum_item + local_item
        return (
            self.users[index],
            self.positive_items[index],
            torch.tensor(negative_item, dtype=torch.long),
        )


class OverlapUserDataset(Dataset):
    def __init__(self, source_csv_path, target_csv_path, overlapped_num_users):
        source_users = pd.read_csv(
            source_csv_path, usecols=["uid"]
        )["uid"].to_numpy(dtype=np.int64, copy=False)
        target_users = pd.read_csv(
            target_csv_path, usecols=["uid"]
        )["uid"].to_numpy(dtype=np.int64, copy=False)
        users = np.intersect1d(source_users, target_users, assume_unique=False)
        self.users = torch.from_numpy(users[users < overlapped_num_users])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index]


def graph_cache_path(data_root, domain, max_neighbors, seed):
    return os.path.join(
        data_root, f"disco_{domain}_graph_m{max_neighbors}_s{seed}.pt"
    )


def load_or_build_domain_graph(
    data_root, data_info, domain, max_neighbors=10, seed=2025
):
    cache_path = graph_cache_path(data_root, domain, max_neighbors, seed)
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    graph = build_domain_graph(
        os.path.join(data_root, f"stage1_train_{domain}.csv"),
        data_info,
        domain,
        max_neighbors=max_neighbors,
        seed=seed,
    )
    torch.save(graph, cache_path)
    return graph
