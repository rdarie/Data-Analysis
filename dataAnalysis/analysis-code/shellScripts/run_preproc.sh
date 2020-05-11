#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J all_preproc_20190123

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-all_preproc_20190123.stdout
#SBATCH -e ../batch_logs/%j-%a-all_preproc_20190123.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
EXP="exp201901231000"
# EXP="exp201901271000"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=2
# once the synchronization has happened once
python3 './preprocNS5.py' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --makeTruncated --maskMotorEncoder
python3 './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
python3 './synchronizeINStoNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP