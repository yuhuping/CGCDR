# About The Project

This is the implementation of the paper--Cluster-Guided Disentangled Representation for Cold-Start Cross-Domain Recommendation

## Requirements：
numpy==1.26.4  
tqdm==4.67.1  
pandas==2.2.3  

## Quick Start
python run.py --Task=Sport_Cloth --alpha=0.001 --beta=0.001


## Complete Dataset:
[https://drive.google.com/file/d/1lb10uo-v_cg2jQHQrde0tJMPXdHvAlem/view?usp=drive_link](https://drive.google.com/file/d/1LBkE0DUIoPL7yxsZmzCABjCN-WymOWk1/view?usp=drive_link)  
The File directory After unzip:   
  CGCDR/  
  ├── data/  
  │   ├── Cloth_Sport/  
  │   │   ├── stage1_test.csv  
  │   │   ├── stage1_train_meta.csv  
  │   │   ├── stage1_train_src.csv  
  │   │   ├── stage1_train_tgt.csv  
  │   │   └── stage1_val.csv  
  │   ├── CD_Movies/...  
  │   ├── Elec_Phone/...  
  │   ├── Sport_Cloth/...  
  │   ├── Movies_CD/...  
  │    ...  
  │   
  ├── models.py  
  ├── run.py  
  ├── trainer.py  
  └── utils.py  



