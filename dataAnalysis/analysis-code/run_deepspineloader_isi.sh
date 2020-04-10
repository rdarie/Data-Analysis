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
#SBATCH -o ../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp202003201200"
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
WINDOW="--window=miniRC"
TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=2"
ANALYSISSELECTOR="--analysisName=emgStretchTime"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda
python --version

python "./exportForDeepSpine.py" --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim --inputBlockName="lfp" --unitQuery="isichoremg" --alignQuery="stimOn"
# python loadSheepDeepSpine.py
