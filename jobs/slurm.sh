#!/bin/bash
#SBATCH -J rnderr
#SBATCH -o logs/job_%A_%a.out
#SBATCH -e logs/job_%A_%a.out
#SBATCH -c 22
#SBATCH --mem=180G

# up to 3 nodes
#SBATCH --array=500-999:167

# to run: sbatch RoundingErrorEsimation/jobs/slurm.sh

bits=12
start=$SLURM_ARRAY_TASK_ID
end=$(( start + 167 ))
range_arg="${start}:${end}"

cd RoundingErrorEstimation
. .venv/bin/activate
python -m appmax batch utkface ${bits}bit -b ${bits} -s gurobi-barrier -i $range_arg -j 21
