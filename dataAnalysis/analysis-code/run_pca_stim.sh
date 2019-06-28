#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J calc_pca

# Specify an output file
#SBATCH -o ../batch_logs/%j-calc_pca.stdout
#SBATCH -e ../batch_logs/%j-calc_pca.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"
# SELECTOR="201901211000-Proprio_minfr"
SELECTOR="201901271000-Proprio_minfr"
# SELECTOR="201901201200-Proprio_minfr"
# python3 './selectUnitsByMeanFR.py' --exp=$EXP --processAll
# python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=5 --window=miniRC --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
python3 './calcPCAinChunks.py' --exp=$EXP --trialIdx=5  --window=miniRC --alignQuery="(stimCat=='stimOn')" --estimatorName="pca_midPeakOnlyStimNoMotion_full" --selector=$SELECTOR
# EXP="exp201901201200_alt"
# SELECTOR="201901201200-Proprio_minfr"
# python3 './calcPCAinChunks.py' --exp=$EXP --trialIdx=1 --window=miniRC --alignQuery="(stimCat=='stimOn')" --estimatorName="pca_midPeakOnlyStimNoMotion_full" --selector=$SELECTOR