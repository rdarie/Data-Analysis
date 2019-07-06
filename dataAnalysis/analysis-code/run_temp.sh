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

# EXP="exp201901211000"
# EXP="exp201901211000"
EXP="exp201812051000"
# EXP="exp201901201200"
# SELECTOR="201901211000-Proprio_minfr"
# SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"
# ESTIMATOR="201901211000-Proprio_pca"

python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=1 --window=short --unitQuery="not((chanName.str.contains('elec'))or(chanName.str.contains('pca')))" --blockName=other --eventName=stimAlignTimes
# python3 './calcAlignedAsigs.py' --exp=$EXP --trialIdx=1 --window=long --unitQuery="not((chanName.str.contains('elec'))or(chanName.str.contains('pca')))" --blockName=other --eventName=stimAlignTimes
