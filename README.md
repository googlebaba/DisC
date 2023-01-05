# DisC
Source code for "NeurIPS2022-Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure"

paper: https://arxiv.org/pdf/2209.14107.pdf

![image](https://github.com/googlebaba/DisC/blob/main/framework.png)

                                                             The framework of DisC
# Contact
Shaohua Fan, Email:fanshaohua@bupt.edu.cn

# Datasets 
Datasets used for Table 1: https://drive.google.com/file/d/1pv_cFKYJxXpT4qJ6jgvNn17MIovZUrhA/view?usp=sharing

Unseen test set for Table 2: https://drive.google.com/file/d/18LE0RnUBksGHsbO0lFtEC0O4jiO7B9_J/view?usp=sharing  # f[0] is the unbiased test set

# Requirements
pip -r requirements.txt

# Running the model
DisC_GCN 

python Disc_gcn_run.py

DisC_Gin

python Disc_gin_run.py

DisC_Gcnii

python Disc_gcnii_run.py

# Reference
@inproceedings{

> author = {Shaohua Fan, Xiao Wang, Yanhu Mo, Chuan Shi, Jian Tang},
 
> title = {Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure},
 
> booktitle = {NeurIPS},

> year = {2022}
}
