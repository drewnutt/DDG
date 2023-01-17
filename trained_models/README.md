# Model Files
This directory contains all of the trained model files that were used in the creation of the manuscript. Each subdirectory contains a different section of the manuscript and the associated model files and their statistics.

Each subdirectory contains a `tar.gz` file that contains the trained PyTorch model files used to generate the statistics that are presented in the manuscript. The statistics presented in the manuscript are also given in the `.csv` file present in each subdirectory.

## Subdirectories
- `ablation` - Ablation study of the different model components (architecture and loss components)
- `addnl_ligs` - Retrospective lead optimization scenario
- `full_train` - Models fully trained on BindingDB Congeneric Series dataset (named `fullytrained_BDB.tar.gz`) and their statistics on the external datasets (also includes models finetuned on the external datasets)
- `LOO-PFAM` - Leave one protein family out cross validation that was used for the **NeurIPS manuscript only**
- `LOO-PFAM_fewshot` - Leave one protein family out cross validation utilized for the main manuscript, includes few shot and no-shot evaluation as described in the manuscript


