#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J align_raw_stim_202101_27_lfp_rig

# Specify an output file
#SBATCH -o ../../batch_logs/align_stim_202101_27_lfp_rig-%a.out
#SBATCH -e ../../batch_logs/align_stim_202101_27_lfp_rig-%a.out

# Request custom resources
#SBATCH --array=1-2

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=1
source shellScripts/run_exp_preamble_202101_27.sh
source shellScripts/run_align_stim_raw_preamble.sh
###
# python -u ./calcAlignedAsigs.py --chanQuery="all" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalSubfolder='None' --signalBlockPrefix='utah' --signalBlockSuffix='analog_inputs' --verbose --exp=$EXP $AMPFIELDNAME $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER
# python -u ./calcAlignedAsigs.py --chanQuery="all" --outputBlockSuffix="lfp" --eventBlockSuffix='epochs' --signalSubfolder='None' --signalBlockPrefix='utah' --verbose --exp=$EXP $AMPFIELDNAME $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER
#
ALIGNQUERY="--alignQuery=stimOn"
echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
python -u ./calcLaplacianFromTriggeredV4.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  