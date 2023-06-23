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
#SBATCH -J s01_analyze_align_all_2021

# Specify an output file
#SBATCH -o ../../batch_logs/s01_analyze_align_all_2021-%a.out
#SBATCH -e ../../batch_logs/s01_analyze_align_all_2021-%a.out

# Request custom resources
#SBATCH --array=1-4

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

LAZINESS="--lazy"
#
ANALYSISFOLDER="--analysisName=hiRes"
#
SPIKEBLOCKSUFFIX="--spikeFileSuffix=mean_subtracted"
#
# SPIKESOURCE="--spikeSource=tdc"
SPIKESOURCE=""
#
RIGSUFFIX="--rigFileSuffix=analog_inputs"
#
BLOCKPREFIX="--sourceFilePrefix=utah"
#
# SLURM_ARRAY_TASK_ID=1
# 
# exps=(201901_25 201901_26 201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
# exps=(201902 201902_04 201902_05)
exps=(202101_20 202101_21 202101_22)
for A in "${exps[@]}"
do
  echo "step 01, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  echo --blockIdx=$SLURM_ARRAY_TASK_ID
  #
  python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
  python -u ./calcMotionAlignTimesV3.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --plotParamHistograms $LAZINESS
  python -u ./calcRefinedStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS --plotting
  python -u ./calcMotionStimAlignTimesV2.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS --plotParamHistograms
  #
  source shellScripts/run_align_stim_preamble.sh
  #
  python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  #
  python -u ./calcAlignedAsigs.py --chanQuery="lfp" --outputBlockSuffix="lfp" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' $VERBOSITY --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  # CAR
  # python -u ./calcRereferencedTriggered.py --inputBlockSuffix="lfp" --unitQuery="lfp" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY --substituteOneChannel
  #
  ALIGNQUERY="--alignQuery=stimOn"
  echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
  python -u ./calcLaplacianFromTriggeredV3.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  python -u ./calcWaveletFeatures.py --inputBlockSuffix="laplace" --unitQuery="csd" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  ALIGNQUERY="--alignQuery=stimOn"
  UNITQUERY="--unitQuery=lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=lfp"
  python -u ./calcTrialOutliersV3.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
  
  source shellScripts/run_align_motion_preamble.sh
  ###
  python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockSuffix="rig" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' --verbose --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  #
  python -u ./calcAlignedAsigs.py --chanQuery="lfp" --outputBlockSuffix="lfp" --eventBlockSuffix='epochs' --signalBlockSuffix='analyze' $VERBOSITY --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER $SIGNALFOLDER $EVENTFOLDER
  # CAR
  # python -u ./calcRereferencedTriggered.py --inputBlockSuffix="lfp" --unitQuery="lfp" --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY --substituteOneChannel
  #
  ALIGNQUERY="--alignQuery=starting"
  echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
  python -u ./calcLaplacianFromTriggeredV3.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  python -u ./calcWaveletFeatures.py --inputBlockSuffix="laplace" --unitQuery="csd" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ALIGNQUERY="--alignQuery=starting"
  UNITQUERY="--unitQuery=lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=lfp"
  python -u ./calcTrialOutliersV3.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
done