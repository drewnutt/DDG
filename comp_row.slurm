#!/bin/bash
#SBATCH -J compute_seqs
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -p any_cpu
#SBATCH -x n201
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

rsync -a cri_bindb_old.txt ${SCRDIR} 

cd $SCRDIR
echo Running on `hostname`
echo PATH=$PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PYTHONPATH=$PYTHONPATH
echo SCRATCH_DIR=$SCRDIR
echo -e  '\n\n'

python /net/dali/home/mscbio/anm329/scripts/compute_row.py --pdbseqs cri_bindb_old.txt -r ${SLURM_ARRAY_TASK_ID} --out row_${SLURM_ARRAY_TASK_ID}.out

mv row_${SLURM_ARRAY_TASK_ID}.out ${SLURM_SUBMIT_DIR}/simi_mat/


