#!/bin/bash
#SBATCH -J regression_model
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=C5

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~dkoes/local/lib/python3.6/site-packages/:$PYTHONPATH

mkdir /scr/drew/

SCRDIR=/scr/drew
rsync -a cache/ /scr/drew/
rsync -a CV_Sets/ /scr/drew/
rsync -a DDG_reg.py /scr/drew/ 

echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRDIR=$SCRDIR
echo -e  '\n\n'

wd=(0 0 0.1 0.01 0.001)
cmd=`python3 DDG_reg.py --ligtr ${SCRDIR}/lig_train1.molcache2  --rectr ${SCRDIR}/rec_train1.molcache2 --trainfile ${SCRDIR}/train_1.txt --ligte ${SCRDIR}/lig_test1.molcache2 --recte ${SCRDIR}/rec_test1.molcache2 --testfile ${SCRDIR}/test_1.txt -wd ${wd[${SLURM_ARRAY_TASK_ID}]}`

#cmd=`python3 DDG_reg.py --ligtr ${SCRDIR}/lig_train1.molcache2  --rectr ${SCRDIR}/rec_train1.molcache2 --trainfile ${SCRDIR}/train_1.txt --ligte ${SCRDIR}/lig_test1.molcache2 --recte ${SCRDIR}/rec_test1.molcache2 --testfile ${SCRDIR}/test_1.txt`
eval cmd
exit
