#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J delsys_synch

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-delsys_synch.out
#SBATCH -e ../../batch_logs/%j-%a-delsys_synch.out

# Specify account details
#SBATCH --account=bibs-dborton-condo
# Request custom resources
#SBATCH --array=3,4

EXP="exp202003201200"

LAZINESS="--lazy"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=4
python3 -u './synchronizeDelsysToNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --plotting