#!/bin/bash
#SBATCH -J make_gnina_file
#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --partition=dept_64
#SBATCH --cpus-per-task=1

SCRDIR=/scr/drew/
models=(crossdock_default2018 crossdock_default2018_1 crossdock_default2018_2 crossdock_default2018_3 crossdock_default2018_4 default2017 dense dense_1 dense_2 dense_3 dense_4 general_default2018 general_default2018_1 general_default2018_2 general_default2018_3 general_default2018_4)
model=${models[$1]}

echo ${model}
python3 GNINA_RUN/make_gnina_file.py -f separated_sets/training_input.txt --model ${model} >> GNINA_RUN/run_gnina_${model}.txt

sbatch  --array=0-962 GNINA_RUN/run_gnina.slurm ${model}
