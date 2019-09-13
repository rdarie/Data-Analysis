#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignFull.stdout
#SBATCH -e ../batch_logs/%j-alignFull.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp201901211000"
# EXP="exp201901271000"
LAZINESS="--lazy"
python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="long" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="long" --chanQuery="rig" --blockName="rig"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="long" --chanQuery="fr" --blockName="fr"
python3 ./calcAlignedRasters.py --exp=$EXP --processAll $LAZINESS --window="long" --chanQuery="raster" --blockName="raster"
# python3 ./plotAlignedNeurons.py --exp=$EXP --processAll $LAZINESS --window="long" --alignQuery="outboundWithStim100HzCW" --rowName="pedalSizeCat"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="short" --chanQuery="fr" --blockName="fr"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="short" --chanQuery="fr_sqrt" --blockName="fr_sqrt"
python3 ./calcAlignedRasters.py --exp=$EXP --processAll $LAZINESS --window="short" --chanQuery="raster" --blockName="raster"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS --window="short" --chanQuery="rig" --blockName="rig"
python3 ./plotAlignedNeurons.py --exp=$EXP --processAll $LAZINESS --window="short" --alignQuery="outboundWithStim100HzCCW"