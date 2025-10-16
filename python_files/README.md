# Python Scripts for Training and Analysis

This directory contains all of the python scripts used for the training and analysis of the &Delta;&Delta;G models  

Directory contents:
- `ddg_reg_ptrain_siamese.py` - Main python training script for &Delta;&Delta;G models
- `models/` - PyTorch model files for the &Delta;&Delta;G models used in the paper

A typical training command looks like:
```bash
python ddg_reg_ptrain_siamese.py --ligtr <training_ligand_molcache2> --ligte <testing_ligand_molcache2> --recte <training_receptor_molcache2> --rectr <testing_receptor_molcache2> --testfile <testing_types_file> --trainfile <training_types_file> --solver adam --tags <wandb_experiment_tags> --lr 0.000367 --epoch 1000 --use_model multtask_latent_def2018 --batch_size 64 --hidden_size 0 --ddg_loss_weight 10
```
`trainfile` and `testfile` are space-separated files with the following columns:
```
<DDG_cls> <DDG> <DG1> <DG2> <receptor_gninatypes> <ligand1_gninatypes> <ligand2_gninatypes>
```
Examples can be found in the `BindingDB_data` directory.

`ligtr` and `ligte` are the training and testing ligand molcache2 files, respectively. `rectr` and `recte` are the training and testing receptor molcache2 files, respectively.
Molcache2 files can be generated using the [create\_caches2.py](https://github.com/gnina/scripts/blob/3a1960c2fc010432234b46dfc08282bdff61764a/create_caches2.py) script.

