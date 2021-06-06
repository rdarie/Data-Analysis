#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J test_train_split_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split_28.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

#   SLURM_ARRAY_TASK_ID=2
source shellScripts/calc_aligned_motion_preamble.sh
#
# suffixes a through e used for the dimensionality calculation#
# suffix f, for RAUC calculations
ITERATOR="--iteratorSuffix=f"
#
ALIGNQUERYTERM="starting"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u './calcTestTrainSplit.py' --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" --selectionName='lfp_CAR_spectral_fa_mahal' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS
python -u './applyTestTrainSplit.py' --resetHDF --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" --selectionName='lfp_CAR_spectral_fa_mahal' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
##
ALIGNQUERYTERM="outbound"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u './calcTestTrainSplit.py' --controlSet --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" --selectionName='lfp_CAR_spectral_fa_mahal' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS
python -u './applyTestTrainSplit.py' --controlSet --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" --selectionName='lfp_CAR_spectral_fa_mahal' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

#
# python -u './applyTestTrainSplit.py' --controlSet --inputBlockSuffix="lfp_CAR_spectral_fa" --unitQuery="factor" --selectionName='lfp_CAR_spectral_fa' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
