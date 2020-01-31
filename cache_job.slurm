#!/bin/bash
#SBATCH -J create_cache
#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --partition=any_cpu
#SBATCH --cpus-per-task=1

export PATH=/net/pulsar/home/dkoes/local18/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local18/lib:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH

python3 ../scripts/create_caches2.py --recmolcache rec_train2.molcache2 --ligmolcache lig_train2.molcache2 -c 2 -d separated_sets CV_Sets/train_2.txt 

python3 ../scripts/create_caches2.py --recmolcache rec_test2.molcache2 --ligmolcache lig_test2.molcache2 -c 2 -d separated_sets CV_Sets/test_2.txt 

python3 ../scripts/create_caches2.py --recmolcache rec_train3.molcache2 --ligmolcache lig_train3.molcache2 -c 2 -d separated_sets CV_Sets/train_3.txt 

python3 ../scripts/create_caches2.py --recmolcache rec_test3.molcache2 --ligmolcache lig_test3.molcache2 -c 2 -d separated_sets CV_Sets/test_3.txt 

