#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J qa_all_202101_21

# Specify an output file
#SBATCH -o ../../batch_logs/qa_all_202101_21-%a.out
#SBATCH -e ../../batch_logs/qa_all_202101_21-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=4
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=1
source shellScripts/run_exp_preamble_202101_21.sh
source shellScripts/run_align_stim_preamble.sh
ALIGNQUERY="--alignQuery=stimOn"
UNITQUERY="--unitQuery=lfp"
INPUTBLOCKNAME="--inputBlockSuffix=lfp"
python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults

source shellScripts/run_align_motion_preamble.sh
ALIGNQUERY="--alignQuery=starting"
UNITQUERY="--unitQuery=lfp"
INPUTBLOCKNAME="--inputBlockSuffix=lfp"
python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
