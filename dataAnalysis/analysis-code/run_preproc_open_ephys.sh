#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1,2,3,4

# Specify a job name:
#SBATCH -J open_ephys_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-open_ephys_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-open_ephys_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# EXP="exp201901271000"
# EXP="exp201812051000"
EXP="exp201901070700"

# python3 ./preprocINS.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID
# python3 ./preprocINSfromSIP.py --exp=$EXP
# python3 ./preprocOpenEphys.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --loadMat
# python3 ./synchronizeOpenEphysToINSSIP.py --exp=$EXP
# python3 ./synchronizeOpenEphysToINS.py --exp=$EXP --trialIdx=1
python3 ./calcTrialAnalysisNix.py --trialIdx=$SLURM_ARRAY_TASK_ID  --exp=$EXP --chanQuery="oechorins" --samplingRate=30000
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs
# python3 ./calcStimAlignTimes.py --processAll --exp=$EXP
# python3 ./calcAlignedAsigs.py --exp=$EXP --processAll --window=RC --lazy --chanQuery="oechorins" --blockName=RC --eventName=stimAlignTimes
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --lazy --window=RC --inputBlockName=RC --unitQuery="oechorins" --alignQuery="stimOnLowRate" --rowName= --colName="program" --colControl="999" --styleName= --hueName="amplitude"
