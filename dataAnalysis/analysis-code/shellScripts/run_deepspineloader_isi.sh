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

# EXP="exp202003201200"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
EXP="exp202006171300"

# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
# WINDOW="--window=miniRC"
WINDOW="--window=XS"
# WINDOW="--window=short"
TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=2"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emg1msec"
ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=emg1msecSmooth"
# UNITSELECTOR="--unitQuery=all"
UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinaloremg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# python3 './calcAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes --chanQuery="isispinaloremg" --blockName="lfp" --verbose  --alignFolderName=stim
# UNITSELECTOR="--unitQuery=isispinaloremg"
# python "./exportForDeepSpine.py" --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim --inputBlockName="emg" $UNITSELECTOR --maskOutlierBlocks --alignQuery="stimOn"
# UNITSELECTOR="--unitQuery=isiemgenv"
python "./calcTargetNoiseCeiling.py" --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim --inputBlockName="emg" $UNITSELECTOR --maskOutlierBlocks --alignQuery="stimOn" --plotting
# python "./calcEpochEffect.py" --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim --inputBlockName="emg" $UNITSELECTOR --maskOutlierBlocks --alignQuery="stimOn" --plotting
python "loadSheepDeepSpine.py"