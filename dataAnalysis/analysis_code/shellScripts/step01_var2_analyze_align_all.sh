#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=32:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=16G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J s01_analyze_align_all_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/s01_analyze_align_all_202101_27-%a.out
#SBATCH -e ../../batch_logs/s01_analyze_align_all_202101_27-%a.out

# Request custom resources
#SBATCH --array=1-2

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

LAZINESS="--lazy"
ANALYSISFOLDER="--analysisName=fullRes"
SPIKEBLOCKSUFFIX="--spikeFileSuffix=mean_subtracted"
SPIKESOURCE=""
#
RIGSUFFIX="--rigFileSuffix=analog_inputs"
BLOCKPREFIX="--sourceFilePrefix=utah"
#
SLURM_ARRAY_TASK_ID=3
# 
# exps=(201901_25 201901_26 201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
# exps=(201902_03 201902_04 201902_05)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 01, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  echo --blockIdx=$SLURM_ARRAY_TASK_ID
  #
  python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
  #
  source shellScripts/run_align_stim_fullRes_preamble.sh
  #
  python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  #
  ALIGNQUERY="--alignQuery=stimOn"
  source shellScripts/run_align_motion_fullRes_preamble.sh
  ###
  python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  #
  ALIGNQUERY="--alignQuery=starting"
done