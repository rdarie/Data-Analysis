#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J ins_synch_2021_01_22

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_synch_2021_01_22.out
#SBATCH -e ../../batch_logs/%j-%a-ins_synch_2021_01_22.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1,2,3

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202011271100"
# EXP="exp202011301200"
# EXP="exp202012111100"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012151200"
# EXP="exp202012171200"
EXP="exp202101051100"
EXP="exp202101061100"
EXP="exp202101111100"
EXP="exp202101141100"
EXP="exp202101191100"
EXP="exp202101201100"
EXP="exp202101211100"
EXP="exp202101221100"
# EXP="exp202101251100"

BLOCKSELECTOR=""
# SLURM_ARRAY_TASK_ID=2

## --showFigures --forceRecalc
python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --lazy --usedTENSPulses
