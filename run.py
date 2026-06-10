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
from disco import DisCo
from disco_trainer import DisCoTrainer
from disco_utils import load_or_build_domain_graph

def setup_logging(task_name, info="", model_name="CGCDR"):
    log_dir = os.path.join("log", "DisCo") if model_name.lower() == "disco" else "log"
    os.makedirs(log_dir, exist_ok=True)
    
    if info:  
        log_filename = os.path.join(log_dir, f"{task_name}_{info}.log")
    else:
        log_filename = os.path.join(log_dir, f"{task_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', 
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CGCDR')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--eval_seed', type=int, default=2027)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stopping_step', type=int, default=20)
    parser.add_argument('--Task', type=str, default='Game_Video')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--All', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--eval_negatives', type=int, default=999)
    parser.add_argument('--num_intents', type=int, default=4)
    parser.add_argument('--graph_neighbors', type=int, default=10)
    parser.add_argument('--random_walk_steps', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--disco_alpha', type=float, default=0.1)
    parser.add_argument('--disco_beta', type=float, default=0.3)
    parser.add_argument('--disco_gamma', type=float, default=0.01)
    parser.add_argument('--disco_lambda', type=float, default=0.3)
    parser.add_argument('--dynamic_neg_sampling', action='store_true')
    args = parser.parse_args()

    logger = setup_logging(args.Task, args.info, args.model)
    
    # Print all arguments
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('./data/'+ args.Task + '/'+'id_info.json', 'r') as f:
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
    elif args.model.lower() == 'disco':
        data_root = os.path.join('data', args.Task)
        logger.info("Building/loading DisCo bipartite graphs")
        source_graph = load_or_build_domain_graph(
            data_root,
            data_info,
            'src',
            max_neighbors=args.graph_neighbors,
            seed=args.seed,
        )
        target_graph = load_or_build_domain_graph(
            data_root,
            data_info,
            'tgt',
            max_neighbors=args.graph_neighbors,
            seed=args.seed + 1,
        )
        model = DisCo(
            num_users=total_num_users,
            source_graph=source_graph,
            target_graph=target_graph,
            embedding_dim=args.emb_dim,
            num_intents=args.num_intents,
            alpha=args.disco_alpha,
            beta=args.disco_beta,
            gamma=args.disco_gamma,
            contrast_weight=args.disco_lambda,
            temperature=args.temperature,
            random_walk_steps=args.random_walk_steps,
            ema_decay=args.ema_decay,
            dropout=args.dropout,
        )
        trainer = DisCoTrainer(model, args, data_info)
        trainer.main()
    else:
        raise ValueError(f"Unknown model: {args.model}")
