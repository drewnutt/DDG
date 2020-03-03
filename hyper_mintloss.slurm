#!/bin/bash
#SBATCH -J hyperparam_optim
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p any_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=C5

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~dkoes/local/lib/python3.6/site-packages/:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate DDG
cd ${SLURM_SUBMIT_DIR}

echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo cur_dir=$PWD
echo -e  '\n\n'
wandb agent --count 1 andmcnutt/DDG_model/4sd26rsd
