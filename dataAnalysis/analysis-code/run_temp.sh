#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J alignTemp

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignTemp.stdout
#SBATCH -e ../batch_logs/%j-alignTemp.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp201812051000"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221200"
# SELECTOR="201901211000-Proprio_minfr"
# SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"
# ESTIMATOR="201901211000-Proprio_pca"

# python3 ./preprocINS.py --exp=$EXP --trialIdx=1
# python3 ./preprocINSfromSIP.py --exp=$EXP
python3 ./preprocOpenEphys.py --exp=$EXP
python3 ./synchronizeOpenEphysToINSSIP.py --exp=$EXP
# python3 ./synchronizeOpenEphysToINS.py --exp=$EXP --trialIdx=1
# python3 ./calcTrialAnalysisNix.py --exp=$EXP --trialIdx=1 --chanQuery=oechorins --samplingRate=30000
# python3 ./calcStimAlignTimes.py --trialIdx=1 --exp=$EXP
python3 ./assembleExperimentData.py --exp=$EXP --processAsigs
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll --window=RC --chanQuery="oechorins" --blockName=RC --eventName=stimAlignTimes
python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window=RC --inputBlockName=RC --unitQuery="oechorins" --alignQuery="stimOn" --rowName= --styleName= --hueName="amplitude"
