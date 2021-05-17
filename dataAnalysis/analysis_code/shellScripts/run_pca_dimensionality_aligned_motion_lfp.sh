#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pca_dimen_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_dimen_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-pca_dimen_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh


BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="lfp_CAR_spectral"
ITERATOR="a"
WINDOW="XL"

python -u './calcSignalDimensionality.py' --estimatorName="fa_db" --debugging --datasetName="${TARGET}_${ITERATOR}_${WINDOW}_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting --showFigures

#
TARGET="lfp_CAR_spectral_fa_db"
# python -u './assembleDataFrames.py' --debugging --iteratorSuffix=$ITERATOR --inputBlockSuffix="${TARGET}_fa_db" --unitQuery="${TARGET}_fa_db" --loadFromFrames --exp=$EXP --window=$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="${TARGET}" --window=$WINDOW $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#
# python -u './calcSignalNovelty.py' --estimatorName="mahal" --debugging --datasetName="${TARGET}_${ITERATOR}_${WINDOW}_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting --showFigures


TARGET="lfp_CAR_spectral_fa_db_mahal"
# python -u './assembleDataFrames.py' --debugging --iteratorSuffix=$ITERATOR --inputBlockSuffix="${TARGET}" --unitQuery="${TARGET}" --loadFromFrames --exp=$EXP --window=$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="${TARGET}" --window=$WINDOW $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
