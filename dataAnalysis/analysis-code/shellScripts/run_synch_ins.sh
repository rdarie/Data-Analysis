#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J ins_synch

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_synch.stdout
#SBATCH -e ../../batch_logs/%j-%a-ins_synch.errout

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=2

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
EXP="exp202010271200"

# BLOCKSELECTOR="--inputBlockSuffix=_full"
BLOCKSELECTOR=""

# SLURM_ARRAY_TASK_ID=3

# python3 -u './synchronizeINStoNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --curateManually
python3 -u './synchronizeINStoNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP $BLOCKSELECTOR --curateManually |& tee "../../batch_logs/${EXP}_Block_${SLURM_ARRAY_TASK_ID}_synch_ins"