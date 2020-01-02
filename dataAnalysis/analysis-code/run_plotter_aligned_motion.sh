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
# EXP="exp201901231000"
# SELECTOR="201901271000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
# OUTLIERSWITCH=""
OUTLIERSWITCH="--maskOutlierTrials"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

for QUERY in outboundM outboundXS outboundS outboundL outboundXL
    do
        for BLOCKNAME in rig
            do
                python3 './plotAlignedAsigs.py' --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery=$BLOCKNAME --alignQuery=$QUERY --rowName="pedalDirection" --verbose $OUTLIERSWITCH
            done
        python3 './plotAlignedNeurons.py' --exp=$EXP --processAll --window="long" --alignQuery=$QUERY --selector=$SELECTOR --rowName="pedalDirection" --verbose $OUTLIERSWITCH
    done