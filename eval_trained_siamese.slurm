#!/bin/bash
#SBATCH -J model_eval
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~dkoes/local/lib/python3.6/site-packages/:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate DDG

SCRDIR=/scr/${USER}/${SLURM_JOB_ID}/
if [ ! -d "$SCRDIR" ] ; then
        mkdir -p $SCRDIR
fi
rsync -a cache/ ${SCRDIR}
rsync -a simi_mat/ ${SCRDIR}
rsync -a DDG_class.py ${SCRDIR} 

echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRDIR=$SCRDIR
echo -e  '\n\n'

testortrain=(train test)
#for evaluating on subsets use this one
cmd=`python3 eval_model.py -rm andmcnutt/DDG_model_Regression/2y7197ea -el cache/clustered_lig_${testortrain[${SLURM_ARRAY_TASK_ID}]}_2.molcache2 -er cache/clustered_rec_${testortrain[${SLURM_ARRAY_TASK_ID}]}_2.molcache2 -ef simi_mat/clustered${testortrain[${SLURM_ARRAY_TASK_ID}]}2.types -t CV_2`
eval cmd
exit
