#!/bin/bash
#SBATCH --job-name=mPrimal
#SBATCH --output=logs/mPrimal_%A_%a.out
#SBATCH --error=logs/mPrimal_%A_%a.err
#SBATCH --array=0-1599
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1

betaQ_values=($(seq 44 4 72))
num_trials=100
num_structured=2               
num_betaQ=${#betaQ_values[@]}  

betaQ_index=$(( SLURM_ARRAY_TASK_ID / (num_trials * num_structured) ))
remainder=$(( SLURM_ARRAY_TASK_ID % (num_trials * num_structured) ))
trial=$(( remainder / num_structured + 1 ))
structured_index=$(( remainder % num_structured ))

betaQ=${betaQ_values[$betaQ_index]}
if [[ $structured_index -eq 0 ]]; then
    structured="True"
else
    structured="False"
fi

mkdir -p ./data/mPrimal240/c8 ./logs
export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH

module load python
source g6k/activate
python3 -u exp_successprob.py c=8 r=29 m=30 sigma=8 q=20000 betaQ=${betaQ} structured=${structured} \
    > ./data/mPrimal240/c8/betaQ${betaQ}_struct${structured}_trial${trial}.txt \
    2>&1