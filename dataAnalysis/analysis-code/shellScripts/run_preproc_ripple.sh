#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J ripple_preproc_20200708_raw

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ripple_preproc_20200708_raw.stdout
#SBATCH -e ../../batch_logs/%j-%a-ripple_preproc_20200708_raw.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3,4

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202003091200"
# EXP="exp202003131100"
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
# EXP="exp202007071300"
# EXP="exp202007081300"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version
#
SLURM_ARRAY_TASK_ID=1
python3 ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --ISI --transferISIStimLog
# python3 ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --previewEncoder