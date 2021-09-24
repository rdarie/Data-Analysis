#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_dual_202102_02

# Specify an output file
#SBATCH -o ../../batch_logs/preproc_dual_202102_02-%a.out
#SBATCH -e ../../batch_logs/preproc_dual_202102_02-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1-3

# EXP="exp201901251000"
# has 1 minirc 2 motionsba
# EXP="exp201901261000"
# has 1-3 motion 4 minirc
# EXP="exp201901271000"
# has 1-4 motion 5 minirc
#
# EXP="exp201901311000"
# has 1-4 motion 5 minirc
# EXP="exp201902010900"
#  has 1-4 motion 5 minirc
# EXP="exp201902021100"
# has 3-5 motion 6 minirc; blocks 1 and 2 were bad;
# EXP="exp201902031100"
# has 1-4 motion 5 minirc;
# EXP="exp201902041100"
# has 1-4 motion 5 minirc;
# EXP="exp201902051100"
# has 1-4 motion
########
EXP="exp202101201100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101211100"
# has 1 minirc 2,3 motion+stim 4 motionOnly
EXP="exp202101221100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101251100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101271100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101281100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202102021100"
# has 1 minirc 2 motion+stim 3 motionOnly


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=3

########### get analog inputs separately to run synchronization, etc
# !! --maskMotorEncoder ignores all motor events outside alignTimeBounds
# python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analogOnly
python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analogOnly --maskMotorEncoder

######### finalize dataset
python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullUnfiltered --chunkSize=700
####
# old
# python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullSubtractMeanUnfiltered --chunkSize=700
# python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=nform --fullSubtractMeanUnfiltered

########### get dataset to run spike extraction on
# python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean --chunkSize=700
# python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean

# python -u ./synchronizeNFormToNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=100