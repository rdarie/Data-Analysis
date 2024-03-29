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
#SBATCH -o ../../batch_logs/%j-plotsStim.out
#SBATCH -e ../../batch_logs/%j-plotsStim.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# EXP="exp202003201200"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202006171300"
# EXP="exp202007011300"
# EXP="exp202007021300"
# EXP="exp202009031500"
EXP="exp202009231400"

# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
# WINDOW="--window=miniRC"
WINDOW="--window=XS"
# WINDOW="--window=XSPre"
# WINDOW="--window=short"

BLOCKSELECTOR="--processAll"
# BLOCKSELECTOR="--blockIdx=2"

# ANALYSISFOLDER="--analysisName=hiRes"
ANALYSISFOLDER="--analysisName=default"
# ANALYSISFOLDER="--analysisName=loRes"

# UNITSELECTOR="--unitQuery=all"
UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinaloremg"
# UNITSELECTOR="--unitQuery=isiemgenvoraccorspinal"

BLOCKSELECTOR="--inputBlockName=emg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# python3 './calcAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $ANALYSISFOLDER --eventName=stimAlignTimes --chanQuery="isispinaloremg" --blockName="lfp" --verbose  --alignFolderName=stim
# UNITSELECTOR="--unitQuery=isispinaloremg"

# python "./exportForDeepSpine.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn"

# WINDOW="--window=XSPre"
# python "./exportForDeepSpine.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $BLOCKSELECTOR $UNITSELECTOR --alignQuery="stimOn" --noStim

# UNITSELECTOR="--unitQuery=isiemgenv"
# python "./calcTargetNoiseCeiling.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim --inputBlockName="emg_clean" $UNITSELECTOR --maskOutlierBlocks --alignQuery="stimOn" --plotting
# python "./calcEpochEffect.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim --inputBlockName="emg" $UNITSELECTOR --maskOutlierBlocks --alignQuery="stimOn" --plotting

python "loadSheepDeepSpine.py"