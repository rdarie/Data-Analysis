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
#SBATCH -o ../batch_logs/%j-tdrFull.stdout
#SBATCH -e ../batch_logs/%j-tdrFull.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# ESTIMATOR="201901211000-Proprio_tdrAcr_long_midPeak"
# BLOCKNAME="tdrAcr"
EXP="exp201901271000"
ESTIMATOR="201901271000-Proprio_tdrAcrGLM_long_midPeak"
BLOCKNAME="tdrAcrGLM"
# python3 ./calcUnitLeastSquaresToAsig.py --exp=$EXP --processAll --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --estimatorName=$BLOCKNAME --verbose --plotting
# python3 ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="fr" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr" --estimatorName=$BLOCKNAME --verbose --plotting
# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --estimator=$ESTIMATOR --lazy --profile --verbose
python3 ./calcUnitLeastSquaresToAsig.py --exp=$EXP --processAll --inputBlockName="fr_sqrt" --secondaryBlockName="rig" --alignQuery="midPeak" --unitQuery="fr_sqrt" --estimatorName=$BLOCKNAME --verbose --plotting
python3 ./evaluateUnitRegressionToAsig.py --exp=$EXP --estimator=$ESTIMATOR --lazy --profile --verbose
# python3 ./applyEstimatorToTriggered.py --exp=$EXP --processAll --window="long" --alignQuery="midPeak" --estimator=$ESTIMATOR --lazy --profile --verbose
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCW" --rowName="pedalSizeCat"