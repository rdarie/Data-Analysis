#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J qa_motion_201901_26

# Specify an output file
#SBATCH -o ../../batch_logs/qa_motion_201901_26-%a.out
#SBATCH -e ../../batch_logs/qa_motion_201901_26-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1-3

# SLURM_ARRAY_TASK_ID=3
source shellScripts/run_exp_preamble_26.sh
source shellScripts/calc_aligned_motion_preamble.sh
# ANALYSISFOLDER="--analysisName=default"
ANALYSISFOLDER="--analysisName=hiRes"
#

UNITQUERY="--unitQuery=lfp"
INPUTBLOCKNAME="--inputBlockSuffix=lfp"
# python -u ./calcTrialOutliers.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults

# calculate spike stats, once outliers do not affect the calculation
# python -u ./calcUnitMeanFR.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --maskOutlierBlocks
# python -u ./calcUnitCorrelation.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --plotting --maskOutlierBlocks
# python -u ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $BLOCKSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $WINDOW $LAZINESS --verbose

# for concatenated files#
BLOCKSELECTOR="--blockIdx=2 --processAll"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_CAR"
UNITQUERY="--unitQuery=lfp"
#
# python -u ./calcTrialOutliers.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --saveResults
