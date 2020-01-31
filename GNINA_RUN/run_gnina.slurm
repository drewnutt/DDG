#!/bin/bash
#SBATCH -J run_gnina
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=C5

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:$LD_LIBRARY_PATH

module load cuda/10.2

if ((${SLURM_ARRAY_TASK_ID}==0))
then
	sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} calculate_gnina_DDG.slurm $1
fi

((start=${SLURM_ARRAY_TASK_ID}*8+1))
((fin=${SLURM_ARRAY_TASK_ID}*8+8))
cmd=`sed -n "${start},${fin}p" GNINA_RUN/run_gnina_$1.txt`
while IFS= read -r line
do
	echo "$line"
	eval $line
done <<< $cmd

