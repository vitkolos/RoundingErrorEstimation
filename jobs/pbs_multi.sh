#!/bin/bash

# to run: qsub RoundingErrorEsimation/jobs/pbs_multi.sh

#PBS -N rnderr_multi

# year: 32 cpus, 34gb
#PBS -l select=1:ncpus=32:mem=40gb:scratch_local=40gb

# h:mm:ss
# california: 0.5*3 hours
# year: 2*3 hours
#PBS -l walltime=6:00:00
DATASET="year"

cd ${PBS_O_WORKDIR}/RoundingErrorEstimation

export TMPDIR=$SCRATCHDIR
trap clean_scratch TERM EXIT

module load python/3.11.11-gcc-10.2.1-555dlyc
source .venv/bin/activate

export GRB_LICENSE_FILE=${PBS_O_WORKDIR}/gurobi.lic

ARGS="-i 0:2000 -s gurobi -j 32"

python -m appmax batch $DATASET 4bit -b 4 $ARGS
python -m appmax batch $DATASET 6bit -b 6 $ARGS
python -m appmax batch $DATASET 8bit -b 8 $ARGS

# cd ~/RoundingErrorEstimation/experiments/year
# zip -rq ~/results/year_batch.zip 4bit 6bit 8bit
