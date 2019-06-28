#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J plotsMotionStim

# Specify an output file
#SBATCH -o ../batch_logs/%j-plotsMotionStim.stdout
#SBATCH -e ../batch_logs/%j-plotsMotionStim.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# SELECTOR="201901211000-Proprio_minfr"
SELECTOR="201901271000-Proprio_minfr"
# SELECTOR="201901201200-Proprio_minfr"

#  python3 './plotAlignedNeurons.py' --exp=$EXP --processAll
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll
#  python3 './plotAlignedNeurons.py' --exp=$EXP --processAll  --window=short --selector=$SELECTOR
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll  --window=short
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=long --blockName=other --unitQuery= --alignQuery="(pedalMovementCat=='outbound')" --rowName= --colName=pedalDirection --colControl=NA --hueControl=NA --hueName=pedalSizeCat
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=short --blockName=other --unitQuery=
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window=long --blockName=pca --unitQuery="(chanName.str.contains('pca'))"
#  python3 './plotAlignedNeurons.py' --exp=$EXP --processAll  --window=short --alignQuery="(pedalMovementCat=='return')"
#  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll  --window=short --alignQuery="(pedalMovementCat=='return')"
