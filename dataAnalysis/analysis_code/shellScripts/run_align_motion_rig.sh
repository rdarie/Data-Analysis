#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J align_motion_20190127_rig

# Specify an output file
#SBATCH -o ../../batch_logs/align_motion_20190127_rig-%a.out
#SBATCH -e ../../batch_logs/align_motion_20190127_rig-%a.out

# Request custom resources
#SBATCH --array=1,2,3,4

# Specify account details
#   SBATCH --account=carney-dborton-condo

#    SLURM_ARRAY_TASK_ID=2
source shellScripts/run_exp_preamble_21.sh
source shellScripts/run_align_motion_preamble.sh
#
python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
# python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="rig" --unitQuery="rig" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS