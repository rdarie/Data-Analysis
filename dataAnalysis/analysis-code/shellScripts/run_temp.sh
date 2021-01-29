#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_20200911

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-preproc_20200911.out
#SBATCH -e ../../batch_logs/%j-%a-preproc_20200911.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202011271100"
# EXP="exp202011301200"
# EXP="exp202012071100"
# EXP="exp202012081200"
# EXP="exp202012091200"
# EXP="exp202012101100"
# EXP="exp202012121100"
# EXP="exp202012151200"
# EXP="exp202012161200"
#1EXP="exp202012171200"
EXP="exp202012181200"


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1

############## troubleshooting waveforms
python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --forSpikeSortingUnfiltered
