#!/bin/bash
#SBATCH -J compute_seqs
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -p any_cpu
#SBATCH --ntasks=1

export PATH=/net/pulsar/home/koes/dkoes/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/net/pulsar/home/koes/dkoes/local/cuda-10.1/lib64:/net/pulsar/home/koes/dkoes/local/lib:/net/antonin/usr/local/cuda-10.0/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~dkoes/local/lib/python3.6/site-packages/:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate ddg_cv

SCRDIR=/scr/${USER}/DDG_CV/
if [ ! -d "$SCRDIR" ] ; then
	mkdir -p $SCRDIR
fi

echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRATCH_DIR=$SCRDIR
echo -e  '\n\n'

cd simi_mat/
python /net/dali/home/mscbio/anm329/scripts/clustering.py --cpickle matrix.pickle -i /net/dali/home/mscbio/anm329/deltadeltaG/separated_sets/training_input.txt -o clustered -v

