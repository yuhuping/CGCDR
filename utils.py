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
        
        # 获取文件名用于进度条描述
        file_name = os.path.basename(csv_path)
        
        # 使用 tqdm 添加进度条
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
    """
    专门用于评估阶段的数据集类
    处理包含目标域正样本列表的测试集
    """
    def __init__(self, csv_path, max_len=20):
        """
        初始化评估数据集
        
        参数:
        csv_path: CSV文件路径
        max_len: 序列最大长度
        """
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.records = []
        
        # 获取文件名用于进度条描述
        file_name = os.path.basename(csv_path)
        
        # 使用 tqdm 添加进度条
        for idx, row in tqdm(self.data.iterrows(), 
                            desc=f"Loading Eval {file_name}", 
                            total=len(self.data),
                            ncols=100):
            try:
                uid = row['uid']
                pos = row['pos_iid']
                neg = row['neg_iid']
                
                # 处理源域序列
                src = []
                if 'pos_seq' in row and pd.notna(row['pos_seq']):
                    src = ast.literal_eval(row['pos_seq'])
                src_ids = [int(i) for i in src][:max_len]
                if len(src_ids) < max_len:
                    src_ids += [0] * (max_len - len(src_ids))
                
                # 处理目标域正样本列表
                tgt_iids = []
                if 'tgt_iids' in row and pd.notna(row['tgt_iids']):
                    tgt_iids = ast.literal_eval(row['tgt_iids'])
                
                # 存储记录
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
    """
    从指定范围采样候选物品ID（包含正样本和负样本）
    
    参数:
    pos_id: 正样本物品ID (标量值)
    pos_ids: 用户目标域历史交互序列 (1D数组)
    MIN: 采样范围最小值
    MAX: 采样范围最大值
    neg_sample_size: 负样本数量
    seed: 随机种子 (保证可复现性)
    
    返回:
    cand_ids: 候选物品ID数组 [1 + neg_sample_size]
    """
    # 创建独立的随机状态对象
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
    
    # 组合正负样本
    cand_ids = np.concatenate([[pos], negs])
    
    return cand_ids

