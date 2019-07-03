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

EXP="exp201901211000"
# EXP="exp201901271000"
SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901271000-Proprio_minfrmaxcorr"
ESTIMATOR="201901211000-Proprio_pca_long_midPeak"
# ESTIMATOR="201901271000-Proprio_pca_long_midPeak"

#  python3 './calcUnitMeanFR.py' --exp=$EXP --processAll --window=long --verbose
#  python3 './calcUnitCorrelation.py' --exp=$EXP --processAll --window=long --verbose --plotting
#  python3 './selectUnitsByMeanFRandCorrelation.py' --exp=$EXP --processAll --window=long
#  python3 './plotAlignedNeurons.py' --exp=$EXP --processAll --window=long --selector=$SELECTOR
#  python3 './calcPCAinChunks.py' --exp=$EXP --processAll --window=long --selector=$SELECTOR --lazy --verbose
#  python3 './applyEstimatorToTriggered.py' --exp=$EXP --processAll --window=long --estimator=$ESTIMATOR --lazy --verbose
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=long --verbose
#  python3 './applyEstimatorToTriggered.py' --exp=$EXP --processAll --window=short --estimator=$ESTIMATOR --lazy --verbose
python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=long --alignQuery=midPeakWithStim --rowName=pedalSizeCat --verbose