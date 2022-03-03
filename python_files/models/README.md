## Model Files
Model files are named with what convolutional architecture they use for their siamese arms, `dense` or `def2018`.

The main models utilized in the paper are `multtask_latent_def2018_model.py` and `multtask_latent_dense_model` which are the Default2018 and Dense Siamese models, respectively.

The following models are used in the ablation study:

Name in Ablation Study | Model File
----- | -----
Standard | `multtask_latent_def2018_model.py`
Concatenation | `multtask_latent_def2018_concat_model.py`
No Siamese Network | `def2018_single_model.py`

