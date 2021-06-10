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

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_CAR_spectral_fa_mahal"
UNITSELECTOR="--unitQuery=mahal"

# ALIGNQUERYTERM="startingNoStim"
ALIGNQUERYTERM="starting"
# ALIGNQUERYTERM="outbound"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

TARGET="lfp_CAR_spectral_fa_mahal"
ITERATOR="f"
#
python -u "./calcSignalRecruitmentV2.py" --iteratorSuffix=ros --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
python -u "./transformIterator.py" --iteratorSuffix=ros --iteratorOutputName=noRos --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --verbose=1
#
# python -u "./calcSignalNoiseCeiling.py" --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
# python -u "./plotSignalRecruitment.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
#
python -u "./calcSignalRecruitmentRegression.py" --estimatorName=enr_refit --iteratorSuffix=noRos --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --verbose=1
python -u "./calcSignalRecruitmentRegression.py" --estimatorName=enr_refit --iteratorSuffix=ros --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --verbose=1
# python -u "./processSignalRecruitmentRegression.py" --estimatorName=enr_refit --iteratorSuffix=noRos --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --verbose=1
#
