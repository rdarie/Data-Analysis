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
# EXP="exp201901271000"
# EXP="exp202010201200"
# EXP="exp202010271200"
EXP="exp202011201100"
EXP="exp202012111100"
EXP="exp202012121100"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1

# UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"

# TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
TRIALSELECTOR="--processAll"

# WINDOW="--window=miniRC"
WINDOW="--window=M"
# WINDOW="--window=XS"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"

OUTLIERMASK="--maskOutlierBlocks"

#STATSOVERLAY="--overlayStats"
TIMEWINDOWOPTS="--winStart=200 --winStop=800"

python3 -u './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $UNITSELECTOR $WINDOW --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="rig" --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="lfp" --unitQuery="lfp" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK