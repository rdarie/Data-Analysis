#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J alignStim_20201217_rig

# Specify an output file
#SBATCH -o ../../batch_logs/%j_%a_alignStim_20201217_rig.stdout
#SBATCH -e ../../batch_logs/%j_%a_alignStim_20201217_rig.errout

# Request custom resources
#SBATCH --array=3

# Specify account details
#SBATCH --account=carney-dborton-condo

####  SLURM_ARRAY_TASK_ID=3
source shellScripts/run_align_stim_preamble.sh

python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockName="rig" --eventBlockName='analyze' --signalBlockName='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER