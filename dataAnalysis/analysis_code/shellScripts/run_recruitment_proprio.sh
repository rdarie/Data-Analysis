#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J mahalanobis_recruitment_lfp_27

# Specify an output file
#SBATCH -o ../../batch_logs/mahalanobis_recruitment_lfp_27.out
#SBATCH -e ../../batch_logs/mahalanobis_recruitment_lfp_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_temp.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
UNITSELECTOR="--unitQuery=mahal"

# ALIGNQUERYTERM="startingNoStim"
ALIGNQUERYTERM="starting"
# ALIGNQUERYTERM="outbound"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
ITERATOR=ma

TARGET="lfp_CAR_spectral_scaled_mahal_ledoit"
INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
#
python -u "./calcSignalRecruitmentV2.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
# python -u "./plotSignalRecruitment.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY

TARGET="lfp_CAR_mahal_ledoit"
INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
#
python -u "./calcSignalRecruitmentV2.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY

TARGET="lfp_CAR_spectral_scaled"
INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
#
python -u "./calcSignalRecruitmentV2.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
python -u "./plotSignalRecruitmentV2.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
#
TARGET="lfp_CAR"
INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
#
python -u "./calcSignalRecruitmentV2.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
python -u "./plotSignalRecruitmentV2.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
