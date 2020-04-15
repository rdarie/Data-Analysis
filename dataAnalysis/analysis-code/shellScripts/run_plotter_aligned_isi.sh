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
# EXP="exp202003091200"
#
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
#
# WINDOW="--window=miniRC"
WINDOW="--window=miniRC"
# WINDOW="--window=short"
# WINDOW="--window=extraShort"
#
TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=2"
#
# ANALYSISSELECTOR="--analysisName=emg"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
# ANALYSISSELECTOR="--analysisName=emgHiRes"
ANALYSISSELECTOR="--analysisName=emg1msec"
#
# UNITSELECTOR="--unitQuery=all"
UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinaloremg"
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda
python --version
#  --maskOutlierBlocks
python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp" --groupPagesBy="electrode" --maskOutlierBlocks --invertOutlierBlocks --individualTraces
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp" --groupPagesBy="electrode" --maskOutlierBlocks --invertOutlierBlocks
python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp" --groupPagesBy="electrode" --maskOutlierBlocks --individualTraces
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp" --groupPagesBy="electrode" --maskOutlierBlocks
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --inputBlockName="lfp" $UNITSELECTOR --alignQuery="stimOff" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --inputBlockName="lfp" $UNITSELECTOR --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --inputBlockName="lfp" $UNITSELECTOR --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides --individualTraces
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --inputBlockName="lfp" $UNITSELECTOR --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides --invertOutlierBlocks --individualTraces

# stim spikes
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $ANALYSISSELECTOR
