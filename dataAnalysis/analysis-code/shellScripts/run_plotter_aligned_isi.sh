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
#SBATCH --account=bibs-dborton-condo

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
# EXP="exp202004271200"
# EXP="exp202004301200"
EXP="exp202005011400"
# EXP="exp202003201200"
#
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
#
# WINDOW="--window=miniRC"
# WINDOW="--window=miniRC"
# WINDOW="--window=short"
# WINDOW="--window=extraExtraShort"
WINDOW="--window=extraShort"
#
TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=3"
# TRIALSELECTOR="--blockIdx=1"
#
# ANALYSISSELECTOR="--analysisName=emg"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
# ANALYSISSELECTOR="--analysisName=emgHiRes"
ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=emg1msec"
# ANALYSISSELECTOR="--analysisName=emg1msecSmooth"
# ANALYSISSELECTOR="--analysisName=emg1msecNoLFPFilterSmoothEMG"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isiemg"

BLOCKSELECTOR="--inputBlockName=emg_clean"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version
# --maskOutlierBlocks --invertOutlierBlocks --individualTraces

#  --maskOutlierBlocks
python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz" --maskOutlierBlocks
python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName="RateInHz" --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --maskOutlierBlocks --enableOverrides --overlayStats
# stim spikes
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR

# #  --maskOutlierBlocks
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
# UNITSELECTOR="--unitQuery=isispinal"
# BLOCKSELECTOR="--inputBlockName=lfp_clean"
# WINDOW="--window=extraExtraShort"
# #
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz"
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName="RateInHz" --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides --overlayStats
# 