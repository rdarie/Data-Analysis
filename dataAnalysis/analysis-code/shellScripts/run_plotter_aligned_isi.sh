#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J plotsStim

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202003201200"
# EXP="exp202006171300"
# EXP="exp202007011300"
EXP="exp202009031500"
# EXP="exp202007021300"
#
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
#
WINDOW="--window=XS"
# WINDOW="--window=XXS"
#
TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=3"
# TRIALSELECTOR="--blockIdx=1"
#
ANALYSISSELECTOR="--analysisName=loRes"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isispinal"
# UNITSELECTOR="--unitQuery=isiemg"
# UNITSELECTOR="--unitQuery=isiemgoracc"
UNITSELECTOR="--unitQuery=isiemgenvoraccorspinal"


BLOCKSELECTOR="--inputBlockName=all"
# BLOCKSELECTOR="--inputBlockName=emg"
# BLOCKSELECTOR="--inputBlockName=emg_clean"
# BLOCKSELECTOR="--inputBlockName=lfp"
# BLOCKSELECTOR="--inputBlockName=emg_clean"
# BLOCKSELECTOR="--inputBlockName=lfp_clean"

# --maskOutlierBlocks --invertOutlierBlocks --individualTraces

# python3 -u './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz" --maskOutlierBlocks
#  --maskOutlierBlocks
python3 -u './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName="electrode" --rowControl= --colName="RateInHz" --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# stim spikes
# python3 -u './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR