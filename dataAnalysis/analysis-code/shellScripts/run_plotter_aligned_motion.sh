#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J plotsMotionStim

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsMotionStim.stdout
#SBATCH -e ../../batch_logs/%j-plotsMotionStim.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# EXP="exp201901231000"
UNITSELECTOR="--selector=_minfrmaxcorr"
# UNITSELECTOR=""
OUTLIERSWITCH=""
# OUTLIERSWITCH="--maskOutlierBlocks"
WINDOW="--window=long"
# TRIALSELECTOR="--blockIdx=2"
TRIALSELECTOR="--processAll"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

#python3 ./calcBlockOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --saveResults --plotting --alignQuery="all" --verbose
#
for QUERY in midPeakM midPeakXS midPeakS midPeakL midPeakXL
    do
        for BLOCKNAME in rig
            do
                python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW --inputBlockName=$BLOCKNAME --unitQuery=$BLOCKNAME --alignQuery=$QUERY --hueName="amplitudeCat" --rowName="pedalDirection" $OUTLIERSWITCH --verbose
            done
        python3 './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR $WINDOW --alignQuery=$QUERY $UNITSELECTOR --hueName="amplitudeCat" --rowName="pedalDirection" $OUTLIERSWITCH --verbose
    done
