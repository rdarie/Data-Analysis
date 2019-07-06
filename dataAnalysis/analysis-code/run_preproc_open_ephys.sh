#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --array=1,2,4,5,7,8,9

# Specify a job name:
#SBATCH -J nsp_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-nsp_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-nsp_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# EXP="exp201901271000"
EXP="exp201812051000"

# python3 './preprocINSfromSIP.py' --exp=$EXP
# python3 './preprocOpenEphys.py' --exp=$EXP
# python3 './synchronizeOpenEphysToINS.py' --exp=$EXP
python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --window=RC --unitQuery="oech" --blockName=other --eventName=stimAlignTimes
python3 './plotAlignedAsigs.py' --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --window=RC --inputBlockName=other --unitQuery="all" --alignQuery="stimOn" --rowName= --hueName="amplitude"


