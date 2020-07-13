#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J delsys_synch

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-delsys_synch.stdout
#SBATCH -e ../../batch_logs/%j-%a-delsys_synch.errout

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1

# EXP="exp202003201200"
# EXP="exp202003191400"
# EXP="exp202004251400"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202003181300"
# EXP="exp202006171300"
EXP="exp202007011300"
# EXP="exp202007021300"

LAZINESS="--lazy"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=1
python3 -u './synchronizeDelsysToNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=1 --plotting