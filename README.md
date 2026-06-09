<div align="center">

# CGCDR

### Cluster-Guided Disentangled Representation for Cold-Start Cross-Domain Recommendation

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Required-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Task-Cross--Domain%20Recommendation-0F766E?style=flat-square" />
  <img src="https://img.shields.io/badge/Setting-Cold%20Start-7C3AED?style=flat-square" />
</p>

<p>
  <img src="https://img.shields.io/badge/Model-CGCDR-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-334155?style=flat-square" />
  <img src="https://img.shields.io/badge/Data-Multi--Task-2563EB?style=flat-square" />
  <img src="https://img.shields.io/badge/Output-Logs%20%2B%20Checkpoints-DB2777?style=flat-square" />
</p>

**PyTorch implementation of CGCDR for cold-start cross-domain recommendation.**

Training pipeline, model definition, dataset layout, logging, and checkpoint saving are all included in this repository.

</div>

---

## Highlights

- Designed for **cold-start users** in **cross-domain recommendation** settings.
- Includes separate training for **source domain**, **target domain**, and **overlap/meta stage**.
- Supports multiple domain pairs such as `Sport_Cloth`, `Game_Video`, and `Movies_CD`.

## Requirements

```bash
pip install torch numpy==1.26.4 pandas==2.2.3 tqdm==4.67.1
```

Recommended environment:

- Python `3.10+`
- PyTorch
- CUDA-enabled GPU if available

## Project Layout

```text
CGCDR/
├── data/
│   ├── Cloth_Sport/
│   ├── CD_Movies/
│   ├── Elec_Phone/
│   ├── Game_Video/
│   ├── Movies_CD/
│   ├── Phone_Elec/
│   ├── Sport_Cloth/
│   └── Video_Game/
├── log/
├── saved/
├── models.py
├── run.py
├── trainer.py
└── utils.py
```

## Dataset

Dataset download:

**[Google Drive](https://drive.google.com/file/d/1LBkE0DUIoPL7yxsZmzCABjCN-WymOWk1/view?usp=drive_link)**

After extraction, organize the files like this:

```text
data/
├── Cloth_Sport/
│   ├── id_info.json
│   ├── stage1_train_src.csv
│   ├── stage1_train_tgt.csv
│   ├── stage1_train_meta.csv
│   ├── stage1_val.csv
│   └── stage1_test.csv
├── CD_Movies/
├── Elec_Phone/
├── Game_Video/
├── Movies_CD/
├── Phone_Elec/
├── Sport_Cloth/
└── Video_Game/
```

## Quick Start

```bash
python run.py --Task=Sport_Cloth --alpha=0.001 --beta=0.001
```

Other available tasks:

`Sport_Cloth` | `Cloth_Sport` | `Game_Video` | `Video_Game` | `Movies_CD` | `CD_Movies` | `Elec_Phone` | `Phone_Elec`

### DisCo

The repository also contains a paper-based reimplementation of **DisCo:
Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain
Recommendation**:

```bash
python run.py --model=DisCo --Task=Sport_Cloth --epoch=100 \
  --emb_dim=128 --num_intents=4 --graph_neighbors=10 \
  --random_walk_steps=3 --disco_beta=0.3 --disco_lambda=0.3
```

On the first run, capped source/target bipartite adjacency tables are built
from the existing stage-1 CSV files and cached under `data/<Task>/`.

## Main Arguments

| Argument | Description |
| --- | --- |
| `--Task` | Dataset task name |
| `--model` | Model name, default is `CGCDR` |
| `--epoch` | Number of training epochs |
| `--lr` | Learning rate |
| `--alpha` | Cluster-related loss weight |
| `--beta` | Contrastive loss weight |
| `--seed` | Random seed |
| `--info` | Extra suffix for log filename |

## Outputs

The training process writes:

- logs to `log/`
- checkpoints to `saved/<Task>/`

## Citation

If this repository is useful for your research, please cite the corresponding CGCDR paper.
