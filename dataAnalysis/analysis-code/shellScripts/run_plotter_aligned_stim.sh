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
#SBATCH --account=carney-dborton-condo

# EXP="exp201901221000"
# EXP="exp201901201200"
# EXP="exp201901261000"
EXP="exp201901271000"
EXP="exp202010201200"
EXP="exp202010271200"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1

# UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
UNITSELECTOR=""

# TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
TRIALSELECTOR="--processAll"

# WINDOW="--window=miniRC"
WINDOW="--window=XS"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"

#  --maskOutlierBlocks
python3 -u './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $UNITSELECTOR $WINDOW --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="rig" --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="utahlfp" --unitQuery="utahlfp" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides