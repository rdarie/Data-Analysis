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
# EXP="exp201901221000"
# SELECTOR="201901211000-Proprio_minfr"
# SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"
# ESTIMATOR="201901211000-Proprio_pca"
EXP="exp201901271000"
TRIALIDX="1"
GLMBLOCKNAME="tdrCmbGLM"
OLSBLOCKNAME="tdrAcr"
GLMESTIMATOR="Block001_${GLMBLOCKNAME}_long_midPeak"
OLSESTIMATOR="Block001_${OLSBLOCKNAME}_long_midPeak"
#
SELECTOR="Block001_minfrmaxcorr"
#
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
# python3 ./calcBlockAnalysisNix.py --exp=$EXP --blockIdx=$TRIALIDX --chanQuery="all"
# python3 ./calcMotionStimAlignTimes.py --exp=$EXP --blockIdx=$TRIALIDX --plotParamHistograms
# python3 ./calcFR.py --exp=$EXP --blockIdx=$TRIALIDX
# python3 ./calcFRsqrt.py --exp=$EXP --blockIdx=$TRIALIDX
# 
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
# python3 ./calcAlignedRasters.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --chanQuery="raster" --blockName="raster"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --chanQuery="fr" --blockName="fr"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --chanQuery="rig" --blockName="rig"
# 
# python3 ./calcUnitMeanFR.py --exp=$EXP --blockIdx=$TRIALIDX --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
# python3 ./calcUnitCorrelation.py --exp=$EXP --blockIdx=$TRIALIDX --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
# python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP --blockIdx=$TRIALIDX --verbose
# 
# python3 ./calcUnitOLSToAsig.py --exp=$EXP --blockIdx=$TRIALIDX --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --estimatorName=$OLSBLOCKNAME --verbose --plotting
# python3 ./evaluateUnitOLSToAsig.py --exp=$EXP --estimator=$OLSESTIMATOR --lazy --profile --verbose
# 
# python3 ./calcUnitGLMToAsig.py --exp=$EXP --blockIdx=$TRIALIDX --selector=$SELECTOR --inputBlockName="raster" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="raster" --estimatorName=$GLMBLOCKNAME --verbose --plotting
# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --estimator=$GLMESTIMATOR --lazy --profile --verbose
# 
python3 ./calcRasterSimilarityMatrix.py --exp=$EXP --processAll --selector=$SELECTOR --alignQuery="midPeak" --unitQuery="raster" --verbose --plotting
# python3 './applyEstimatorToTriggered.py' --exp=$EXP --blockIdx=$TRIALIDX --window="short" --alignQuery="outboundWithStim100HzCCW" --estimator=$ESTIMATOR --lazy --profile
# 
# python3 ./calcAlignedRasters.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --chanQuery="raster" --blockName="raster"
# python3 ./plotAlignedNeurons.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="long" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="short" --chanQuery="fr" --blockName="fr"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="short" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
# python3 ./calcAlignedRasters.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="short" --chanQuery="raster" --blockName="raster"
# python3 ./plotAlignedNeurons.py --exp=$EXP --blockIdx=$TRIALIDX --lazy --window="short" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
