#!/bin/bash
#SBATCH -J calc_gnina_DDG
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=C5

echo $1
num=89598
for file in $(ls slurm-${num}_*.out)
do
	grep 'separated_sets\|Affinity\|CNNa\|CNNs\|Intramolecular' $file >> GNINA_RUN/all_output_$1.txt
done

lines=$(grep -c 'Affinity' GNINA_RUN/all_output_$1.txt)
echo $lines
if (($lines==7704))
then
	rm slurm-${num}_*.out
fi

python3 GNINA_RUN/calculate_gnina_DDG.py -gf GNINA_RUN/all_output_$1.txt --trainf separated_sets/training_input.txt --model $1
