#!/bin/bash
#SBATCH -J making_gninatype
#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --partition=dept_24
#SBATCH --cpus-per-task=1

cd separated_sets/

export PATH=/net/pulsar/home/koes/dkoes/local18/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local18/lib:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH

which gninatyper
a=$(sed -n "${SLURM_ARRAY_TASK_ID}p" rec_list.txt)
cd "$a/"

for file in *.mol2
do
	gninatyper $file ${file%.mol2}
done 
