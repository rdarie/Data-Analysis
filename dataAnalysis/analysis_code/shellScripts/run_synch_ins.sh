#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J ins_synch_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/ins_synch_20190127-%a.out
#SBATCH -e ../../batch_logs/ins_synch_20190127-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1,2,3,4

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

EXP="exp201901261000"
# EXP="exp201901271000"

# EXP="exp202101061100"
# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"

BLOCKSELECTOR=""
# BLOCKSELECTOR="--inputINSBlockSuffix="
SLURM_ARRAY_TASK_ID=4

## --showFigures --forceRecalc
# python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --lazy --usedTENSPulses --forceRecalc
python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --showFigures --lazy --curateManually --forceRecalc
