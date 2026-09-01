#!/bin/bash

# run like this: qsub -v bits=6 RoundingErrorEsimation/jobs/pbs.sh

#PBS -N rnderr
#PBS -o logs/rnderr/
#PBS -e logs/rnderr/

# year: 2gb per job
# utkface: 10gb per job
#PBS -l select=1:ncpus=16:mem=160gb:scratch_local=160gb

# year: 5 minutes
# utkface: 23 hours
# h:mm:ss
#PBS -l walltime=23:00:00

# 8 at once (we have 16 licenses), step 16 (= jobs)
#PBS -J 0-1999:16%8
jobs=16
start=$PBS_ARRAY_INDEX
end=$(( start + jobs ))
range_arg="${start}:${end}"

cd ${PBS_O_WORKDIR}/RoundingErrorEstimation

export TMPDIR=$SCRATCHDIR
trap clean_scratch TERM EXIT

module load python/3.11.11-gcc-10.2.1-555dlyc
source .venv/bin/activate

export GRB_LICENSE_FILE=${PBS_O_WORKDIR}/gurobi.lic

python -m appmax batch utkface ${bits}bit -b ${bits} -s gurobi-barrier -i $range_arg -j $jobs
