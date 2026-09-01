#!/bin/bash

# run like this: qsub -v dataset=year RoundingErrorEsimation/jobs/pbs_multi.sh

#PBS -N rnderr_multi
#PBS -o logs/rnderr_multi/
#PBS -e logs/rnderr_multi/

# year: 32 cpus, 40gb
#PBS -l select=1:ncpus=32:mem=40gb:scratch_local=40gb

# h:mm:ss
# california: 0.5*3 hours
# year: 2*3 hours
#PBS -l walltime=6:00:00

cd ${PBS_O_WORKDIR}/RoundingErrorEstimation

export TMPDIR=$SCRATCHDIR
trap clean_scratch TERM EXIT

module load python/3.11.11-gcc-10.2.1-555dlyc
source .venv/bin/activate

export GRB_LICENSE_FILE=${PBS_O_WORKDIR}/gurobi.lic

args="-i 0:2000 -s gurobi -j 32"

python -m appmax batch $dataset 4bit -b 4 $args
python -m appmax batch $dataset 6bit -b 6 $args
python -m appmax batch $dataset 8bit -b 8 $args

# cd ~/RoundingErrorEstimation/experiments
# zip -rq ~/results/combined.zip california/*bit_multi year/*bit_multi utkface/*bit_new
