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
# EXP="exp201901231000"
UNITSELECTOR="--selector=_minfrmaxcorr"
# UNITSELECTOR=""
# OUTLIERSWITCH=""
OUTLIERSWITCH="--maskOutlierTrials"
WINDOW="--window=long"
# TRIALSELECTOR="--trialIdx=2"
TRIALSELECTOR="--processAll"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

# python3 ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --saveResults --plotting --inputBlockName="fr" --alignQuery="all" --unitQuery="fr" --verbose
#
for QUERY in outboundXS outboundS outboundM outboundL outboundXL
    do
        for BLOCKNAME in rig
            do
                python3 './plotAlignedAsigs.py' --exp=$EXP $TRIALSELECTOR $WINDOW --inputBlockName=$BLOCKNAME --unitQuery=$BLOCKNAME --alignQuery=$QUERY --rowName="pedalDirection" $OUTLIERSWITCH --verbose
            done
        # python3 './plotAlignedNeurons.py' --exp=$EXP $TRIALSELECTOR $WINDOW --alignQuery=$QUERY $UNITSELECTOR --rowName="pedalDirection" $OUTLIERSWITCH --verbose
    done
