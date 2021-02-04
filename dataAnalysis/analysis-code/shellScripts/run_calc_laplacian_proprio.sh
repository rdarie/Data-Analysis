#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J analysis_calc_2021_01_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-analysis_calc_2021_01_28.out
#SBATCH -e ../../batch_logs/%j-%a-analysis_calc_2021_01_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
EXP="exp202101281100"

LAZINESS="--lazy"
#
# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"
#
module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=2
#
python -u ./calcLaplacian.py --chanQuery="lfp" --inputBlockSuffix='analyze' --verbose --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS
