#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J analysis_calc_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-analysis_calc_20190127.stdout
#SBATCH -e ../../batch_logs/%j-%a-analysis_calc_20190127.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3,4

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202012121100"
EXP="exp202012171200"
EXP="exp202101061100"

LAZINESS="--lazy"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"

# BLOCKSUFFIX="--sourceFileSuffix=mean_subtracted"
BLOCKSUFFIX="--spikeFileSuffix=mean_subtracted"

RIGSUFFIX="--rigFileSuffix=analog_inputs"
# BLOCKSUFFIX=""

BLOCKPREFIX="--sourceFilePrefix=utah"
module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=3
# python -u ./synchronizeSIMItoNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
# python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $BLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
##
# python -u ./calcMotionStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --plotParamHistograms $LAZINESS
python -u ./calcMotionAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID  $ANALYSISFOLDER --plotParamHistograms $LAZINESS
# python -u ./calcStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --plotParamHistograms $LAZINESS
##
# python -u ./calcFR.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER
# python -u ./calcFRsqrt.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID