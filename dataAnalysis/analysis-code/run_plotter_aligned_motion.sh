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

EXP="exp201901211000"
# EXP="exp201901271000"
SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901271000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfrmaxcorr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda
python --version

for BLOCKNAME in fr
    do
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="rig" --alignQuery="midPeakCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="rig" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery=$BLOCKNAME --alignQuery="midPeakM_CW" --rowName="pedalSizeCat"
        #  python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery=$BLOCKNAME --alignQuery="midPeakM_CCW" --rowName="pedalSizeCat"
        python3 './plotAlignedNeurons.py' --exp=$EXP --processAll --window="long" --alignQuery="outboundM_CW" --rowName="pedalSizeCat"
        python3 './plotAlignedNeurons.py' --exp=$EXP --processAll --window="long" --alignQuery="outboundM_CCW" --rowName="pedalSizeCat"
    done