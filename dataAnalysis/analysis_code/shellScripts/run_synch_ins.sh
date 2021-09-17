#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J ins_synch_201901_31

# Specify an output file
#SBATCH -o ../../batch_logs/ins_synch_201901_31-%a.out
#SBATCH -e ../../batch_logs/ins_synch_201901_31-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp201901221000"
# has 1 minirc 2-3 motion
# EXP="exp201901231000"
# has 1 motion
# EXP="exp201901240900"
# has 1 minirc 2 motion
# EXP="exp201901251000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp201901281200"
# has 1-4 motion
# EXP="exp201901301000"
# has 1-3 motion 4 minirc
EXP="exp201901311000"
# has 1-4 motion 5 minirc

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
# SLURM_ARRAY_TASK_ID=2

## --showFigures --forceRecalc
#
# python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --lazy --usedTENSPulses --forceRecalc
python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --lazy --curateManually --forceRecalc --showFigures

# for BLOCKIDX in 1 2 3 4
# do
#     python -u './synchronizeINStoNSP_stimBased.py' --blockIdx=$BLOCKIDX --exp=$EXP $BLOCKSELECTOR --inputNSPBlockSuffix=analog_inputs --addToNIX --lazy --curateManually --forceRecalc --showFigures |& tee "../../batch_logs/${EXP}_Block_${BLOCKIDX}_synch_ins"
# done