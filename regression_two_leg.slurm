#!/bin/bash
#SBATCH -J same_rot
#SBATCH -t 5-0:00:00
#SBATCH -N 1
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~dkoes/local/lib/python3.6/site-packages/:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate DDG

SCRDIR=/scr/${USER}/${SLURM_JOB_ID}/
if [ ! -d "$SCRDIR" ] ; then
	mkdir -p $SCRDIR
fi

rsync -a cache ${SCRDIR}
rsync -a simi_mat ${SCRDIR}
rsync -a ddg_reg_prev.py ${SCRDIR} 
cd $SCRDIR
echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRATCH_DIR=$SCRDIR
echo -e  '\n\n'

hid=(0 8 16 32 64 128)
python3 ddg_reg_prev.py --ligtr cache/clustered_lig_train_0.molcache2 --ligte cache/clustered_lig_test_0.molcache2 --recte cache/clustered_rec_test_0.molcache2 --rectr cache/clustered_rec_train_0.molcache2 --testfile simi_mat/clusteredtest0.types --trainfile simi_mat/clusteredtrain0.types --epoch 300 --hidden_size ${hid[${SLURM_ARRAY_TASK_ID}]} --lr 0.001
#below is trying to emulate paper
#python3 ddg_reg_prev.py --ligtr cache/lig_train4.molcache2 --ligte cache/lig_val4.molcache2 --recte cache/rec_val4.molcache2 --rectr cache/rec_train4.molcache2 --testfile CV_Sets/val_4_group.txt --trainfile CV_Sets/train_4_group.txt --solver adam --module_depth 2 --module_filters 32 --filter_factor 1 --num_modules 2 --tags paper_sorta
