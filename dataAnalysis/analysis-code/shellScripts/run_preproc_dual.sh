#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_dual_2021_02_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-preproc_dual_2021_02_20.out
#SBATCH -e ../../batch_logs/%j-%a-preproc_dual_2021_02_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# EXP="exp202101141100"
# EXP="exp202101191100"
EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1

########### get analog inputs separately to run synchronization, etc
# !! --maskMotorEncoder ignores all motor events outside alignTimeBounds
# python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analogOnly --maskMotorEncoder

######### finalize dataset
# python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullUnfiltered --chunkSize=700
# python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=nform --fullSubtractMeanUnfiltered

########### get dataset to run spike extraction on
python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean --chunkSize=700
# python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean

# python -u ./synchronizeNFormToNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=100