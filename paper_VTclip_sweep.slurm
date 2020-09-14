#!/bin/bash
#SBATCH -J clip_sweep
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

LASTLIG=${SCRDIR}separated_sets/2VJ7/2VJ702.sdf
if [ ! -f "$LASTLIG" ] ; then
	rsync -a sep_sets_ob_conv.tar.gz ${SCRDIR}
	tar xzf  ${SCRDIR}sep_sets_ob_conv.tar.gz -C ${SCRDIR}
else
	echo "$LASTLIG exists"
fi
rsync -a CV_Sets ${SCRDIR}
rsync -a python_files/reg_emu_paper.py ${SCRDIR} 
cd $SCRDIR
echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRATCH_DIR=$SCRDIR
echo -e  '\n\n'

wandb agent --count 1 andmcnutt/DDG_model_Regression/vu6oy4a5
