#!/bin/bash
#SBATCH -J rnderr
#SBATCH -o logs/job_%A_%a.out
#SBATCH -e logs/job_%A_%a.out
#SBATCH -c 22
#SBATCH --mem=180G

# up to 3 nodes
#SBATCH --array=500-999:167

# to run: sbatch RoundingErrorEsimation/jobs/slurm.sh

BITS=12
START=$SLURM_ARRAY_TASK_ID
END=$(( START + 167 ))
RANGE_ARG="${START}:${END}"

cd RoundingErrorEstimation
. .venv/bin/activate
python -m appmax batch utkface ${BITS}bit -b ${BITS} -s gurobi-barrier -i $RANGE_ARG -j 21
