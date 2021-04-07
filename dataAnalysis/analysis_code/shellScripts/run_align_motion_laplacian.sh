#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=300G

# Specify a job name:
#SBATCH -J csd_calc_motion_2021_02_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-csd_calc_motion_2021_02_20.out
#SBATCH -e ../../batch_logs/%j-%a-csd_calc_motion_2021_02_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_align_motion_preamble.sh

python -u ./calcLaplacianFromTriggered.py --plotting --useKCSD --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="kcsd" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="kcsd" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
