#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J qa_motion_2021_01_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-qa_motion_2021_01_20.out
#SBATCH -e ../../batch_logs/%j-%a-qa_motion_2021_01_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=2,3

#  SLURM_ARRAY_TASK_ID=3
source shellScripts/calc_aligned_motion_preamble.sh
ANALYSISFOLDER="--analysisName=default"
UNITQUERY="--unitQuery=lfp"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_CAR"

# python -u ./calcTrialOutliers.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
# python -u ./calcTrialOutliersPCA.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults

# calculate spike stats, once outliers do not affect the calculation
# python -u ./calcUnitMeanFR.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --maskOutlierBlocks
# python -u ./calcUnitCorrelation.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --plotting --maskOutlierBlocks
# python -u ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $BLOCKSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $WINDOW $LAZINESS --verbose

# for concatenated files
BLOCKSELECTOR="--blockIdx=2 --processAll"
python -u ./calcTrialOutliers.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
