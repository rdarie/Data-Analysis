#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J tdrFull

# Specify an output file
#SBATCH -o ../../batch_logs/%j-tdrFull.out
#SBATCH -e ../../batch_logs/%j-tdrFull.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# ESTIMATOR="201901211000-Proprio_tdrAcr_long_midPeak"
# BLOCKNAME="tdrAcr"
EXP="exp201901271000"
GLMBLOCKNAME="tdrCmbGLM"
OLSBLOCKNAME="tdrAcr"
GLMESTIMATOR="201901271000-Proprio_${GLMBLOCKNAME}_long_midPeak"
OLSESTIMATOR="201901271000-Proprio_${OLSBLOCKNAME}_long_midPeak"
SELECTOR="201901271000-Proprio_minfrmaxcorr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version
# python3 ./calcUnitLeastSquaresToAsig.py --exp=$EXP --processAll --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --estimatorName=$BLOCKNAME --verbose --plotting
# python3 ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="fr" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr" --estimatorName=$BLOCKNAME --verbose --plotting
# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --estimator=$ESTIMATOR --lazy --profile --verbose
# 
python3 ./calcUnitMeanFR.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./calcUnitCorrelation.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP --processAll --verbose
# 
# python3 ./calcUnitOLSToAsig.py --exp=$EXP --processAll --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --estimatorName=$OLSBLOCKNAME --verbose --plotting
# python3 ./evaluateUnitOLSToAsig.py --exp=$EXP --estimator=$OLSESTIMATOR --lazy --profile --verbose
#
python3 ./calcUnitGLMToAsig.py --exp=$EXP --processAll --selector=$SELECTOR --inputBlockName="raster" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="raster" --estimatorName=$GLMBLOCKNAME --verbose
python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --estimator=$GLMESTIMATOR --lazy --profile --verbose
#
# python3 ./applyEstimatorToTriggered.py --exp=$EXP --processAll --window="long" --alignQuery="midPeak" --estimator=$ESTIMATOR --lazy --profile --verbose
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCW" --rowName="pedalSizeCat"