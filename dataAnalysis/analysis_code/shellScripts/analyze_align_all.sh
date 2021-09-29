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
#SBATCH -J analyze_align_all_201902_03_lfp_rig

# Specify an output file
#SBATCH -o ../../batch_logs/analyze_align_all_201902_03_lfp_rig-%a.out
#SBATCH -e ../../batch_logs/analyze_align_all_201902_03_lfp_rig-%a.out

# Request custom resources
#SBATCH --array=1-5

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

source ./shellScripts/run_exp_preamble_201902_03.sh
echo --blockIdx=$SLURM_ARRAY_TASK_ID

python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
python -u ./calcMotionAlignTimesV3.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --plotParamHistograms $LAZINESS
python -u ./calcRefinedStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS --plotting
python -u ./calcMotionStimAlignTimesV2.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS --plotParamHistograms

# source ./shellScripts/run_exp_preamble_201902_03.sh
source shellScripts/run_align_stim_preamble.sh
###

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
python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults

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
python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
