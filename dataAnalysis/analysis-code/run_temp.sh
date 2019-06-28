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

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"
# SELECTOR="201901211000-Proprio_minfr"
SELECTOR="201901271000-Proprio_minfr"
# SELECTOR="201901201200-Proprio_minfr"

#  python3 './calcMotionStimAlignTimes.py' --exp=$EXP --trialIdx=3
#  python3 './calcFR.py' --exp=$EXP --trialIdx=3
#  python3 './calcFRsqrt.py' --exp=$EXP --trialIdx=3
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=3 --chanQuery="(not(chanName.str.contains('elec')or(chanName.str.contains('pca')))" --blockName=other
#  python3 './calcAlignedAsigsOneUnit.py' --exp=$EXP --trialIdx=3 --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=3 --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=3 --chanQuery="(not(chanName.str.contains('elec')or(chanName.str.contains('pca')))" --blockName=other
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=3 --window=long --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=3 --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
#  python3 './applyEstimatorToTriggered.py' --exp=$EXP --trialIdx=3 --estimator=Trial003_pca
#  python3 './calcAlignedRasters.py' --exp=$EXP --trialIdx=3 --chanQuery="(chanName.str.endswith('raster'))" --blockName=raster
#  python3 './selectUnitsByMeanFR.py' --exp=$EXP --trialIdx=3 --window=short --verbose
#  python3 './selectUnitsByMeanFRandCorrelation.py' --exp=$EXP --trialIdx=3 --window=short --verbose
#  python3 './calcPCAinChunks.py' --exp=$EXP --trialIdx=3 --window=short --selector=Trial003_minmeanfr --verbose
#  python3 './saveRasterForGPFA.py' --exp=$EXP --trialIdx=5 --window=miniRC --selector=$SELECTOR --alignQuery="" --alignSuffix=stim
#  python3 './saveRasterForGPFA.py' --exp=$EXP --trialIdx=5 --window=miniRC --selector=$SELECTOR --alignQuery="(amplitude==0)" --alignSuffix=noStim
#  python3 './plotAsigsAlignedToMotionStim.py' --exp=$EXP --trialIdx=3 --window=short --blockName=gpfa --chanQuery="(chanName.str.contains('gpfa'))" --alignQuery="(pedalMovementCat=='midPeak')"
#  python3 './applyGPFAtoTriggered.py' --exp=$EXP --trialIdx=3 --window=short --selector=$SELECTOR --verbose
#  python3 './optimizeGPFAdimensions.py' --exp=$EXP --trialIdx=3 --window=short --selector=$SELECTOR --verbose
#  python3 './plotGPFAoptimization.py' --exp=$EXP --runIdx=1
python3 './plotPCA3D.py' --exp=$EXP --processAll --window=long
