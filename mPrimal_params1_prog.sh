#!/bin/bash
#SBATCH --job-name=mPrimal
#SBATCH --output=logs/mPrimal_%A_%a.out
#SBATCH --error=logs/mPrimal_%A_%a.err
#SBATCH --array=0-499
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1

num_structured=2 
num_trials=250   

trial=$(( SLURM_ARRAY_TASK_ID / num_structured + 1 ))
structured_index=$(( SLURM_ARRAY_TASK_ID % num_structured ))

if [[ $structured_index -eq 0 ]]; then
    structured="True"
else
    structured="False"
fi

mkdir -p ./data/mPrimal240/c3 ./logs
export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH

module load python
source g6k/activate

python3 -u exp_successprob.py c=3 r=59 m=60 sigma=11 q=48000 progressive=True tours=2 max_betaQ=80 structured=${structured} \
    > ./data/mPrimal240/c3/prog_struct${structured}_trial${trial}.txt \
    2>&1