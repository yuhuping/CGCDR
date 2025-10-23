import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import json

import logging
import os

from models import *
from trainer import *

def setup_logging(task_name, info=""):
    # 创建logs目录如果不存在
    if not os.path.exists('log'):
        os.makedirs('log')
    
    # 设置日志文件名（添加info参数）
    if info:  # 如果info不为空，添加到文件名中
        log_filename = f"log/{task_name}_{info}.log"
    else:
        log_filename = f"log/{task_name}.log"
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',  # 去掉毫秒部分
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stopping_step', type=int, default=20)
    parser.add_argument('--Task', type=str, default='Sport_Cloth')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--All', type=bool, default=False, help='是否使用全部模型跑')
    parser.add_argument('--alpha', type=float, default=0.01, help='聚类学习损失权重')
    parser.add_argument('--beta', type=float, default=0.001, help='CGCDR对比学习权重')
    # 添加info参数用于日志文件名
    parser.add_argument('--info', type=str, default='', help='日志文件命名信息')
    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(args.Task, args.info)
    
    # Print all arguments
    # 打印所有参数
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('./data/ready/'+ args.Task + '/'+'id_info.json', 'r') as f:
            data_info = json.load(f)
    total_num_users = data_info['total_num_users']
    total_num_items = data_info['total_num_items']
    
    if args.model == 'CGCDR':
        
        alpha = args.alpha
        logger.info('CGCDR alpha: {}'.format(alpha))
        beta = args.beta
        logger.info('CGCDR beta: {}'.format(beta))

        src_num_clusters = 256
        tgt_num_clusters = 256
        
        if args.Task == 'Movies_CD':
            src_num_clusters = 384
        # if args.Task == 'CD_Movies':
        #     tgt_num_clusters = 384
        if args.Task == 'Sport_Cloth':
            src_num_clusters = 128
            tgt_num_clusters = 128
        model = CGCDR(num_users=total_num_users, num_items=total_num_items+1, emb_dim=64, data_info=data_info, src_num_clusters=src_num_clusters, tgt_num_clusters=tgt_num_clusters, alpha=alpha, beta=beta).cuda()
        trainer = CGCDRTrainer(model,args,data_info)
        trainer.main()