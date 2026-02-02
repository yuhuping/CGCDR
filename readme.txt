Requirements：
numpy==1.26.4
tqdm==4.67.1
pandas==2.2.3
=================================================================
This is the implementation of the paper--Cluster-Guided Disentangled Representation for Cold-Start Cross-Domain Recommendation

Step1: unzip ./data/ready/Game_Video/GV.zip -d ./data/ready/Game_Video
Step2: python run.py --Task=Game_Video --alpha=0.001 --beta=0.001

Note: The complete dataset will be available after acceptance.

=================================================================
Complete Dataset:
https://drive.google.com/file/d/1lb10uo-v_cg2jQHQrde0tJMPXdHvAlem/view?usp=drive_link
The Structure After unzip：
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
|
├── models.py
├── run.py
├── trainer.py
└── utils.py
