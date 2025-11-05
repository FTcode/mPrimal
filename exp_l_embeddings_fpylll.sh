#!/bin/bash
#SBATCH --job-name=lembed
#SBATCH --output=logs/lembed%A_%a.out
#SBATCH --error=logs/lembed%A_%a.err
#SBATCH --array=0-1599
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1

# ---- Mapping array index to (l, trial) ----
l_values=(1 2 3 4 5 6 7 8)
num_l=${#l_values[@]}
num_trials=200

l_index=$(( SLURM_ARRAY_TASK_ID / num_trials ))
trial=$(( SLURM_ARRAY_TASK_ID % num_trials + 1 ))

l=${l_values[$l_index]}

mkdir -p ./data/lembed/fpylll ./logs
export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH

module load python
source g6k/activate

python3 -u exp_successprob.py \
    c=15 r=14 m=15 q=2100 sigma=4 betaQ=58 tours=10 structured=False use_fpylll=True l=${l} \
    > ./data/lembed/fpylll/c15_l${l}_trial${trial}.txt \
    2>&1