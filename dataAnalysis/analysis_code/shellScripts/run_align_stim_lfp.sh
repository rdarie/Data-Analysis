#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J align_stim_2021_01_28_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j_%a_align_stim_2021_01_28_lfp.out
#SBATCH -e ../../batch_logs/%j_%a_align_stim_2021_01_28_lfp.out

# Request custom resources
#SBATCH --array=1,2,3

# Specify account details
#SBATCH --account=carney-dborton-condo

# SLURM_ARRAY_TASK_ID=1
source shellScripts/run_align_stim_preamble.sh
###
python -u ./calcAlignedAsigs.py --chanQuery="lfp" --outputBlockSuffix="lfp" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' $VERBOSITY --exp=$EXP $AMPFIELDNAME $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
# python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="lfp" --unitQuery="lfp" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
#
python -u ./calcRereferencedTriggered.py --inputBlockSuffix="lfp" --unitQuery="lfp" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY --substituteOneChannel
# python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY