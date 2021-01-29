#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J delsys_synch_0903

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-delsys_synch_0903.out
#SBATCH -e ../../batch_logs/%j-%a-delsys_synch_0903.errout

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=1,2,3,4

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
# EXP="exp202008180700"
# EXP="exp202009031500"

LAZINESS="--lazy"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=4
python3 -u './synchronizeDelsysToNSP.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=2 --plotting