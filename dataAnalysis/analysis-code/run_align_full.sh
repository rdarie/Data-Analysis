#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J alignFull

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignFull.stdout
#SBATCH -e ../batch_logs/%j-alignFull.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

#  root /gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code
#  EXP="exp201901211000_alt"
EXP="exp201901271000_alt"

#  python3 './assembleExperimentData.py' --exp=$EXP
#  python3 './calcMotionStimAlignTimes.py' --trialIdx=1 --exp=$EXP --processAll --plotParamHistograms
#  python3 './calcFR.py' --trialIdx=1 --exp=$EXP --processAll
#  python3 './calcFRsqrt.py' --trialIdx=1 --exp=$EXP --processAll
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --chanQuery="(not(chanName.str.contains('elec')))" --blockName=other
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
#  python3 './calcAlignedRasters.py' --exp=$EXP --processAll --chanQuery="(chanName.str.endswith('raster'))" --blockName=raster
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --window=long --chanQuery="(not(chanName.str.contains('elec')))" --blockName=other
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --window=long --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --processAll --window=long --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
#  python3 './calcAlignedRasters.py' --exp=$EXP --processAll --window=long --chanQuery="(chanName.str.endswith('raster'))" --blockName=raster
#  python3 './selectUnitsByMeanFR.py' --exp=$EXP --processAll
