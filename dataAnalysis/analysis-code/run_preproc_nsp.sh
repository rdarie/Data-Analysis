#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J nsp_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-nsp_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-nsp_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1

# EXP="exp201901070700"
# EXP="exp201901201200"
EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda
python --version

SLURM_ARRAY_TASK_ID=3
# python3 ./preprocNS5.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --makeTruncated
python3 ./preprocNS5.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --makeFull
