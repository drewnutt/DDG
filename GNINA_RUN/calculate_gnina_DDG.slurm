#!/bin/bash
#SBATCH -J calc_gnina_DDG
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=C5


for file in (ls slurm-${SLURM_JOB_DEPENDENCY//![0-9]/}_*.out)
do
	grep 'separated_sets\|Affinity\|CNN\|Intramolecular' $file >> RUN_GNINA/all_output_$1.txt
done

((lines=grep -c all_output_$1.txt))
if ((lines==7704*5))
then
	rm slurm-${SLURM_JOB_DEPENDENCY//![0-9]/}_*.out
fi

python3 calculate_gnina_DDG.py -gf RUN_GNINA/all_output_$1.txt --trainf separated_sets/training_input --model $1
