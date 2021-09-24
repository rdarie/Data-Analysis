#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J analysis_calc_hr_202101_21

# Specify an output file
#SBATCH -o ../../batch_logs/analysis_calc_hf_202101_21-%a.out
#SBATCH -e ../../batch_logs/analysis_calc_hf_202101_21-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1-3
#SBATCH --export=CCV_HEADLESS=1
#
LAZINESS="--lazy"
#
# ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=hiRes"
ANALYSISFOLDER="--analysisName=hiResHiFreq"
# ANALYSISFOLDER="--analysisName=normalizedByImpedance"
# ANALYSISFOLDER="--analysisName=default"
#
SPIKEBLOCKSUFFIX="--spikeFileSuffix=mean_subtracted"
#
# SPIKESOURCE="--spikeSource=tdc"
SPIKESOURCE=""
#
RIGSUFFIX="--rigFileSuffix=analog_inputs"
# BLOCKSUFFIX=""
#
BLOCKPREFIX="--sourceFilePrefix=utah"
#

# SLURM_ARRAY_TASK_ID=1

source ./shellScripts/run_exp_preamble_202101_21.sh
echo --blockIdx=$SLURM_ARRAY_TASK_ID

python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
# or, if has kinematics
# python -u ./synchronizeSIMItoNSP_stimBased.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --showFigures --inputBlockSuffix=analog_inputs  --inputBlockPrefix=utah --plotting --forceRecalc
# python -u ./calcProprioAnalysisNix.py --hasKinematics --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy

# python -u ./calcMotionAlignTimesV3.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID  $ANALYSISFOLDER --plotParamHistograms $LAZINESS

# python -u ./calcRefinedStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS --plotting
# or, if wanting to trim down the dataset (e.g. for time consuming uperations such as CSD calculation, can remove the stimOff labels
# python -u ./calcRefinedStimAlignTimes.py --removeLabels='stimOff' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS

# python -u ./calcMotionStimAlignTimesV2.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS --plotParamHistograms

# python -u ./calcFR.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER
# python -u ./calcFRsqrt.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID

# RC
# python -u ./calcProprioAnalysisNix.py --hasEMG --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
# python -u ./calcRefinedStimAlignTimes.py --disableRefinement --removeLabels='stimOff' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS
