#!/bin/bash

# to run (for 8-bit): qsub -v BITS=8 RoundingErrorEsimation/jobs/pbs.sh

#PBS -N rnderr
#PBS -o logs/job_^array_index^.out
#PBS -e logs/job_^array_index^.err

# year: 2gb per job
# utkface: 10gb per job
#PBS -l select=1:ncpus=16:mem=160gb:scratch_local=160gb

# year: 5 minutes
# utkface: 15 hours
# h:mm:ss
#PBS -l walltime=23:00:00

# 10 at once (we have 16 licenses), step 16 (= JOBS)
#PBS -J 0-1999:16%10
JOBS=16
START=$PBS_ARRAY_INDEX
END=$(( START + JOBS ))
RANGE_ARG="${START}:${END}"

cd ${PBS_O_WORKDIR}/RoundingErrorEstimation

export TMPDIR=$SCRATCHDIR
trap clean_scratch TERM EXIT

module load python/3.11.11-gcc-10.2.1-555dlyc
source .venv/bin/activate

export GRB_LICENSE_FILE=${PBS_O_WORKDIR}/gurobi.lic

# python -m appmax batch california ${BITS}bit -b ${BITS} -s gurobi -i $RANGE_ARG -j $JOBS
# python -m appmax batch year ${BITS}bit -b ${BITS} -s gurobi -i $RANGE_ARG -j $JOBS
python -m appmax batch utkface ${BITS}bit -b ${BITS} -s gurobi-barrier -i $RANGE_ARG -j $JOBS
