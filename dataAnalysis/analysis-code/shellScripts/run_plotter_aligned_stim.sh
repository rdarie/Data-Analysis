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
# UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
UNITSELECTOR=""
# SELECTOR="_minfrmaxcorr"
TRIALSELECTOR="--blockIdx=4"
# TRIALSELECTOR="--processAll"
WINDOW="--window=miniRC"
ANALYSISNAME="--analysisName=loRes"
# ANALYSISNAME="--analysisName=default"
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

#  --maskOutlierBlocks
python3 './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR $ANALYSISNAME $UNITSELECTOR $WINDOW --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISNAME $WINDOW --inputBlockName="rig" --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
# python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $ANALYSISNAME $WINDOW --inputBlockName="utahlfp" --unitQuery="utahlfp" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides