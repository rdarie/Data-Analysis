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
# EXP="exp201812051000"
EXP="exp201901070700"

# python3 './preprocINS.py' --exp=$EXP --trialIdx=1
# python3 './preprocINSfromSIP.py' --exp=$EXP
# python3 './preprocOpenEphys.py' --exp=$EXP
# python3 './synchronizeOpenEphysToINSSIP.py' --exp=$EXP
# python3 './synchronizeOpenEphysToINS.py' --exp=$EXP --trialIdx=1
python3 ./calcTrialAnalysisNix.py --exp=$EXP --trialIdx=1 --chanQuery=oechorins --samplingRate=30000
# python3 './assembleExperimentData.py' --exp=$EXP --processAsigs
# python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --window=RC --chanQuery="oech" --blockName=RC --eventName=stimAlignTimes
# python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=RC --inputBlockName=RC --unitQuery="oech" --alignQuery="stimOn" --rowName= --styleName= --hueName="amplitude"
