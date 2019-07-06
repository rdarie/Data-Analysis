#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J plotsStim

# Specify an output file
#SBATCH -o ../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# EXP="exp201901271000"
# EXP="exp201901201200"
EXP="exp201812051000"
# SELECTOR="201901201200-Proprio_minfr"
MINIRCIDX="1"

python3 './plotAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX  --window=short --inputBlockName=other --unitQuery="all" --alignQuery="stimOn" --rowName= --hueName="amplitude"
# python3 './plotNeuronsAlignedToStim.py' --exp=$EXP --trialIdx=$MINIRCIDX  --window=miniRC  --selector=$SELECTOR
