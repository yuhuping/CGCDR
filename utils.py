import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random   
import pandas as pd
import argparse
import json
import os
import ast
from tqdm import tqdm

class SeqItemDataset(Dataset):
    def __init__(self, csv_path, max_len=20):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.records = []

        file_name = os.path.basename(csv_path)
        
        for idx, row in tqdm(self.data.iterrows(), 
                            desc=f"Loading {file_name}", 
                            total=len(self.data),
                            ncols=100):

            uid = row['uid']
            pos = row['pos_iid']
            neg = row['neg_iid']
            try:
                src = ast.literal_eval(row['pos_seq']) if 'pos_seq' in row else []
                src_ids = [int(i) for i in src][:max_len]
                if len(src_ids) < max_len:
                    src_ids += [0] * (max_len - len(src_ids))
                self.records.append((uid, src_ids, pos, neg))
            except:
                continue

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        uid, src, pos, neg = self.records[idx]
        return torch.tensor(uid), torch.tensor(src), torch.tensor(pos), torch.tensor(neg)


class TestSeqItemDataset(Dataset):
    def __init__(self, csv_path, max_len=20):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.records = []
        
        file_name = os.path.basename(csv_path)

        for idx, row in tqdm(self.data.iterrows(), 
                            desc=f"Loading Eval {file_name}", 
                            total=len(self.data),
                            ncols=100):
            try:
                uid = row['uid']
                pos = row['pos_iid']
                neg = row['neg_iid']
                
                src = []
                if 'pos_seq' in row and pd.notna(row['pos_seq']):
                    src = ast.literal_eval(row['pos_seq'])
                src_ids = [int(i) for i in src][:max_len]
                if len(src_ids) < max_len:
                    src_ids += [0] * (max_len - len(src_ids))

                tgt_iids = []
                if 'tgt_iids' in row and pd.notna(row['tgt_iids']):
                    tgt_iids = ast.literal_eval(row['tgt_iids'])

                self.records.append((uid, src_ids, pos, neg, tgt_iids))
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        uid, src, pos, neg, tgt_iids = self.records[idx]
        return (
            torch.tensor(uid), 
            torch.tensor(src), 
            torch.tensor(pos), 
            torch.tensor(neg),
            torch.tensor(tgt_iids) if tgt_iids else torch.tensor([])
        )

def sample_candidates(pos_id, pos_ids, MIN, MAX, neg_sample_size, seed=None):

    rng = np.random.RandomState(seed)
    # print(f'pos_id: {pos_id}, src_ids: {src_ids}, MIN: {MIN}, MAX: {MAX}, neg_sample_size: {neg_sample_size}')
    pos = int(pos_id)
    exclude_ids = set(int(i) for i in pos_ids)

    all_candidates = np.arange(MIN, MAX + 1)
    mask = ~np.isin(all_candidates, list(exclude_ids))
    valid_candidates = all_candidates[mask]

    negs = rng.choice(
        valid_candidates, 
        size=neg_sample_size,
        replace=False
    )

    cand_ids = np.concatenate([[pos], negs])
    
    return cand_ids

