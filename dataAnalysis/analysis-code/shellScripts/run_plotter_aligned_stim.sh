#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plotsStim

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901221000"
# EXP="exp201901201200"
EXP="exp201901271000"
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
TRIALSELECTOR="--blockIdx=5"
# TRIALSELECTOR="--processAll"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

#  --maskOutlierBlocks
python3 './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR --selector=$SELECTOR --window="long" --unitQuery="all" --alignQuery="stimOn" --rowName= --hueName="amplitude" --alignFolderName=stim --enableOverrides --maskOutlierBlocks
python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR --window="long" --inputBlockName="rig" --unitQuery="all" --alignQuery="stimOn" --rowName= --hueName="amplitude" --alignFolderName=stim --enableOverrides --maskOutlierBlocks
