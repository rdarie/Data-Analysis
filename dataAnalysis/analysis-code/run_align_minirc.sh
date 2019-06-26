#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignMiniRC

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignMiniRC.stdout
#SBATCH -e ../batch_logs/%j-alignMiniRC.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901201200_alt"
# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
MINIRCIDX="5"

#  python3 './calcFR.py' --trialIdx=$MINIRCIDX --exp=$EXP
#  python3 './calcFRsqrt.py' --trialIdx=$MINIRCIDX --exp=$EXP
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --eventName=stimAlignTimes --chanQuery="(not(chanName.str.contains('elec')or(chanName.str.contains('pca')))" --blockName=other
#  python3 './calcStimAlignTimes.py' --trialIdx=$MINIRCIDX --exp=$EXP
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
#  python3 './calcAlignedRasters.py' --exp=$EXP --trialIdx=$MINIRCIDX --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('raster'))" --blockName=raster
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --window=miniRC --eventName=stimAlignTimes --chanQuery="(not(chanName.str.contains('elec')or(chanName.str.contains('pca')))" --blockName=other
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --window=miniRC --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('fr'))" --blockName=fr
#  python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=$MINIRCIDX --window=miniRC --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('fr_sqrt'))" --blockName=fr_sqrt
python3 './calcAlignedRasters.py' --exp=$EXP --trialIdx=$MINIRCIDX --window=miniRC --eventName=stimAlignTimes --chanQuery="(chanName.str.endswith('raster'))" --blockName=raster
