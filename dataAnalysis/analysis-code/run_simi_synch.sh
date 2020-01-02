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
#SBATCH -o ../batch_logs/%j-%a-ins_synch.stdout
#SBATCH -e ../batch_logs/%j-%a-ins_synch.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo
# Request custom resources
#SBATCH --array=1

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
EXP="exp201901271000"


module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

# SLURM_ARRAY_TASK_ID=1
python3 './synchronizeSIMItoNSP.py' --trialIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP