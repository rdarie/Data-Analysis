#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=100:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J ins_preproc_noSlotsOnAndOffRateDelayV4

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_preproc_noSlotsOnAndOffRateDelayV4.out
#SBATCH -e ../../batch_logs/%j-%a-ins_preproc_noSlotsOnAndOffRateDelayV4.out

# Specify account details
#SBATCH --account=carney-dborton-condo


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
EXP="exp202101281100"

SLURM_ARRAY_TASK_ID=2

# --makePlots to make quality check plots
# --showPlots to interactively display quality check plots
# --disableStimDetection to use HUT derived stim start times

ANALYSISFOLDER="--analysisName=default"
# OUTSUFFIX=noSlotsOnAndOffRateDelay
# OUTSUFFIX=noSlotsOnAndOffRateDelayV2
# OUTSUFFIX=noSlotsOnAndOffRateDelayV3
OUTSUFFIX=noSlotsOnAndOffRateDelayV4
# OUTSUFFIX=noSlotsOnRateDelay
# OUTSUFFIX=noSlotsNoRateDelay
# OUTSUFFIX=highThresSlotsOnAndOffRateDelay
# OUTSUFFIX=highThresSlotsOnRateDelay
# OUTSUFFIX=lowThresSlotsOnAndOffRateDelay
# OUTSUFFIX=lowThresSlotsOnAndOffRateDelayV2
# OUTSUFFIX=lowThresSlotsOnAndOffRateDelayV3
# OUTSUFFIX=lowThresSlotsOnAndOffRateDelayV4
# OUTSUFFIX=lowThresSlotsOnRateDelay

python -u './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --makePlots --verbose --outputSuffix=$OUTSUFFIX

LAZINESS='--lazy'
## --showFigures --forceRecalc
python -u ./synchronizeINStoNSP_stimBased.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --outputINSBlockSuffix=$OUTSUFFIX --inputINSBlockSuffix=$OUTSUFFIX --inputNSPBlockSuffix=analog_inputs --addToNIX $LAZINESS --usedTENSPulses --forceRecalc

python -u ./calcRefinedStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analysisName=$OUTSUFFIX --inputINSBlockSuffix=$OUTSUFFIX --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS
