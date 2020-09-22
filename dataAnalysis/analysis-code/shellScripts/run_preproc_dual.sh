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
#SBATCH -o ../../batch_logs/%j-%a-preproc_20200911.stdout
#SBATCH -e ../../batch_logs/%j-%a-preproc_20200911.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=3

# EXP="exp201804271016"
# EXP="exp201805231100"
# EXP="exp201805071032"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202008281100"
# EXP="exp202008311100"
# EXP="exp202009021100"
# EXP="exp202009071200"
# EXP="exp202009101200"
# EXP="exp202009111100"
EXP="exp202009211200"


module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=1

python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --forSpikeSorting
python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --forSpikeSorting

# !! maskMotorEncoder ignores all motor events outside alignTimeBounds
# python3 -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullSubtractMean --maskMotorEncoder
# python3 -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=nform --rippleNForm

# python3 -u ./synchronizeNFormToNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=100
