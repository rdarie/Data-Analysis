#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J ins_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-ins_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-ins_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2

# EXP="exp201901070700"
EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"

python3 './preprocINS.py' --trialIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
# python3 './preprocINS.py' --trialIdx=1 --exp=$EXP --showPlots
