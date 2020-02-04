# Pipeline for GNINA DDG baseline comparison
This pipeline uses the built in model of GNINA to calculate the affinity of a given protein+ligand pair. Then compares all of the ligands for a given receptor with each other to determine deltadeltaG values for each permutation of ligand pairs for a given receptor. All of those statistics are then put together and compared with the training data for the DDG model. This provides a baseline statistic for how GNINA can do without training to predict DDG.
### How to Run Pipeline:
1. start off pipeline by executing `sbatch gninacomp_pipeline.slurm` from the main directory
2. gninacomp\_pipeline.slurm will then create a file where every line is of the form`gnina -r separated_sets/REC/REC.mol2 -l separated_sets/REC/LIG.sdf --score_only --gpu --cnn_scoring --cnn MODEL` where REC is the PDBID of the receptor, LIG is the name of the ligand mol2 file and MODEL is one of the built-in GNINA models
3. gninacomp\_pipeline.slurm will submit a new job as such `sbatch  --array=0-962 GNINA_RUN/run_gnina.slurm MODEL` to run GNINA on all of the ligands
4. A job will be submitted by the first array job to run `sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} GNINA_RUN/calculate_gnina_DDG.slurm MODEL` to calculate all of the DDG for all of the ligand pairs and then calculate the accuracy of the model in comparison to the trianing data. Output accuracy will be in the slurm file in markdown format.
5. The execution will continue with the next model for steps 2-5 and so on until the last model is reached.


Note: errors in running GNINA are put into a file `problem_file.txt`
