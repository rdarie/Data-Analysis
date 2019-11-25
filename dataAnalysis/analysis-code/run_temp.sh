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
EXP="exp201901211000"
# EXP="exp201901221200"
# EXP="exp201901271000"
# SELECTOR="_minfr"
# SELECTOR="_minfrmaxcorr"
TRIALIDX="2"
GLMBLOCKNAME="tdrCmbGLM"
OLSBLOCKNAME="tdrAcr"
GLMESTIMATOR="Trial00${TRIALIDX}_${GLMBLOCKNAME}_long_midPeak"
OLSESTIMATOR="Trial00${TRIALIDX}_${OLSBLOCKNAME}_long_midPeak"
#
SELECTOR="Trial00${TRIALIDX}_minfrmaxcorr"
#
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version
#
# python3 ./calcTrialAnalysisNix.py --exp=$EXP --trialIdx=$TRIALIDX --chanQuery="all"
# python3 ./calcMotionStimAlignTimes.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --plotParamHistograms
# python3 ./calcFR.py --exp=$EXP --trialIdx=$TRIALIDX
# python3 ./calcFRsqrt.py --exp=$EXP --trialIdx=$TRIALIDX
# 
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
# python3 ./calcAlignedRasters.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="raster" --blockName="raster"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr" --blockName="fr"
# python3 ./calcAlignedAsigs.py --exp=$EXP --trialIdx=$TRIALIDX --lazy --window="long" --chanQuery="rig" --blockName="rig"
#
# python3 ./calcUnitMeanFR.py --exp=$EXP --trialIdx=$TRIALIDX --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
# python3 ./calcUnitCorrelation.py --exp=$EXP --trialIdx=$TRIALIDX --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting
# python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP --trialIdx=$TRIALIDX --verbose
# python3 ./calcTrialOutliers.py --exp=$EXP --trialIdx=$TRIALIDX --selector=$SELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting
# python3 ./calcUnitGLMToAsig.py --exp=$EXP --trialIdx=$TRIALIDX --selector=$SELECTOR --inputBlockName="raster" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="raster" --estimatorName=$GLMBLOCKNAME --verbose --plotting
python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --estimator=$GLMESTIMATOR --lazy --profile --verbose