#!/bin/bash
#SBATCH -J calc_gnina_DDG
#SBATCH -t 24:00:00
#SBATCH --partition=any_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

models=(crossdock_default2018 crossdock_default2018_1 crossdock_default2018_2 crossdock_default2018_3 crossdock_default2018_4 default2017 dense dense_1 dense_2 dense_3 dense_4 general_default2018 general_default2018_1 general_default2018_2 general_default2018_3 general_default2018_4)
model=${models[${SLURM_ARRAY_TASK_ID}]}
echo 'Rerunning DDG calculation'
echo $model
#num=102220
#for file in $(ls slurm-${num}_*.out)
#do
#	grep 'separated_sets\|Affinity\|CNNa\|CNNs\|Intramolecular' $file >> GNINA_RUN/all_output_$1.txt
#done
#
#lines=$(grep -c 'Affinity' GNINA_RUN/all_output_$1.txt)
#echo $lines
#if (($lines==7704))
#then
#	rm slurm-${num}_*.out
#fi

python3 GNINA_RUN/calculate_gnina_DDG.py -gf GNINA_RUN/better_all_output_$model.txt --trainf separated_sets/training_input.txt --model $model --vina
