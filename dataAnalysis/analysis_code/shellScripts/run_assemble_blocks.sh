#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignFull_20190127.out
#SBATCH -e ../../batch_logs/%j-alignFull_20190127.out

# Specify account details
#SBATCH --account=carney-dborton-condo
source shellScripts/run_exp_preamble.sh
source shellScripts/run_align_stim_preamble.sh

python -u ./assembleExperimentData.py --exp=$EXP $ANALYSISFOLDER --processAsigs --processRasters
