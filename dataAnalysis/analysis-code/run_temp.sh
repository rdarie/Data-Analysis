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

# EXP="exp201812051000"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221200"
# SELECTOR="201901211000-Proprio_minfr"
# SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"
# ESTIMATOR="201901211000-Proprio_pca"
EXP="exp201901271000"
TRIALIDX="1"
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr" --blockName="fr"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="rig" --blockName="rig"
python3 ./calcUnitRegressionToAsig.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --verbose
# 
# python3 ./calcAlignedRasters.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="raster" --blockName="raster"
# python3 ./plotAlignedNeurons.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="short" --chanQuery="fr" --blockName="fr"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="short" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
# python3 ./calcAlignedRasters.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="short" --chanQuery="raster" --blockName="raster"
# python3 ./plotAlignedNeurons.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="short" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
