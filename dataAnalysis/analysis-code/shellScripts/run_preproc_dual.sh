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
#SBATCH --array=2,3

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
EXP="exp202012171200"
# EXP="exp202012181200"


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=2

# python3 -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --usedTENSPulses

############## init spike sorting
# python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --forSpikeSorting
python -u ./tridesclousCCV.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --arrayName=utah --sourceFileSuffix=spike_preview --remakePrb --removeExistingCatalog --initCatalogConstructor
##
# python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --forSpikeSorting
# python -u ./tridesclousCCV.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --arrayName=nform --sourceFileSuffix=spike_preview --remakePrb --removeExistingCatalog --initCatalogConstructor

########### get dataset to run spike extraction on
# python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean
# python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean

######### finalize dataset
# !! maskMotorEncoder ignores all motor events outside alignTimeBounds
# python3 -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullSubtractMeanWithSpikes
# python3 -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=nform --fullSubtractMeanWithSpikes

# python3 -u ./synchronizeNFormToNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=100
