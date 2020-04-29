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

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
EXP="exp202004271200"
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
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isiemg"

BLOCKSELECTOR="--inputBlockName=emg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda
python --version

# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp_clean" --groupPagesBy="electrode"
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName="RateInHz" --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName="RateInHz" --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides --individualTraces
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides --invertOutlierBlocks --individualTraces

#  --maskOutlierBlocks
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode" --maskOutlierBlocks --invertOutlierBlocks --individualTraces
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode" --maskOutlierBlocks --invertOutlierBlocks
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode" --maskOutlierBlocks --individualTraces
python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --styleName='RateInHz' --sizeName='RateInHz' --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode"
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $WINDOW $UNITSELECTOR $ANALYSISSELECTOR --alignQuery="stimOn" --alignFolderName=stim $BLOCKSELECTOR --groupPagesBy="electrode" --maskOutlierBlocks

# stim spikes
# python3 './plotRippleStimSpikeReport.py' --exp=$EXP $TRIALSELECTOR $ANALYSISSELECTOR
