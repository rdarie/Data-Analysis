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

#$SLURM_ARRAY_TASK_ID
# EXP="exp201901211000"
# EXP="exp201901271000"
# EXP="exp201812051000"
EXP="exp201901070700"
RCTRIALIDX="2"
# python3 ./preprocINS.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID
# python3 ./preprocINSfromSIP.py --exp=$EXP
# python3 ./preprocOpenEphys.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --loadMat
# python3 ./synchronizeOpenEphysToINSSIP.py --exp=$EXP
# python3 ./synchronizeOpenEphysToINS.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID
# python3 ./synchronizeOpenEphysToINS.py --exp=$EXP --trialIdx=$RCTRIALIDX --showPlots
# python3 './preprocNS5.py' --exp=$EXP --trialIdx=$RCTRIALIDX --makeTruncated
# python3 ./synchronizeOpenEphysToNSP.py --exp=$EXP --trialIdx=2 --showPlots
# python3 ./calcTrialAnalysisNix.py --trialIdx=$RCTRIALIDX  --exp=$EXP --chanQuery="all" --suffix=fast
# python3 ./calcFR.py --trialIdx=$RCTRIALIDX --exp=$EXP --suffix=fast
# python3 ./calcStimAlignTimes.py --trialIdx=$RCTRIALIDX --exp=$EXP --suffix=fast
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$RCTRIALIDX --window=RC --lazy --eventName=stimAlignTimes --chanQuery=fr --blockName=fr --suffix=fast
# python3 ./calcAlignedRasters.py --exp=$EXP --trialIdx=$RCTRIALIDX --window=RC --lazy --eventName=stimAlignTimes --chanQuery=raster --blockName=raster --suffix=fast
python3 ./plotAlignedNeurons.py --exp=$EXP --trialIdx=$RCTRIALIDX --window=RC --lazy --alignQuery="stimOnLowRate" --rowName= --colName="electrode" --colControl="control" --styleName= --hueName="amplitude"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$RCTRIALIDX --window=RC --lazy --chanQuery="oechorins" --blockName=RC --eventName=stimAlignTimes --suffix=fast
# python3 ./plotAlignedAsigs.py --exp=$EXP --trialIdx=$RCTRIALIDX --window=RC --lazy --inputBlockName=RC --unitQuery="oechorins" --alignQuery="stimOnLowRate" --rowName= --colName="program" --colControl="999" --styleName= --hueName="amplitude"
