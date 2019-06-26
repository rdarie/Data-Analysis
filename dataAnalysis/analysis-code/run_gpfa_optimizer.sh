#!/bin/bash

# Request runtime:
#SBATCH --time=48:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J optimize_gpfa

# Specify an output file
#SBATCH -o ../batch_logs/%j-optimize_gpfa.stdout
#SBATCH -e ../batch_logs/%j-optimize_gpfa.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"
# SELECTOR="201901211000-Proprio_minfr"
SELECTOR="201901271000-Proprio_minfr"
# SELECTOR="201901201200-Proprio_minfr"

# python3 './selectUnitsByMeanFR.py' --exp=$EXP --processAll
# python3 './selectUnitsByMeanFRandCorrelationFast.py' --exp=$EXP --processAll --verbose
# python3 './optimizeGPFAdimensions.py' --exp=$EXP --processAll --alignSuffix=midPeak
python3 './plotGPFAoptimization.py' --exp=$EXP --processAll --alignSuffix=midPeak
# python3 './applyGPFAtoTriggered.py' --exp=$EXP --processAll --alignQuery="(amplitudeCat==0)" --alignSuffix=midPeakNoStim

# python3 './optimizeGPFAdimensions.py' --exp=$EXP --trialIdx=1 --window=miniRC --alignSuffix=stim
# python3 './plotGPFAoptimization.py' --exp=$EXP --trialIdx=1 --window=miniRC --alignSuffix=stim
# python3 './applyGPFAtoTriggered.py' --exp=$EXP --trialIdx=1 --window=miniRC --alignSuffix=stim