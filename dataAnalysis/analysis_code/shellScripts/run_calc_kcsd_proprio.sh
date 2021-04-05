#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=48:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J laplacian_calc_2021_01_20_kcsd

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-laplacian_calc_2021_01_20_kcsd.out
#SBATCH -e ../../batch_logs/%j-%a-laplacian_calc_2021_01_20_kcsd.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"

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

# SLURM_ARRAY_TASK_ID=2
#
python -u ./calcLaplacianFromAsig.py --useKCSD --inputBlockSuffix='analyze' --chanQuery="lfp" --outputBlockSuffix="kcsd" --eventBlockSuffix='epochs' --verbose --plotting --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS
