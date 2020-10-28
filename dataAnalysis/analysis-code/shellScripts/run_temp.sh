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
#SBATCH -o ../../batch_logs/%j-alignMiniRC.stdout
#SBATCH -e ../../batch_logs/%j-alignMiniRC.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

./shellScripts/run_preproc_ins.sh
./shellScripts/run_analysis_maker_ins_only.sh
./shellScripts/run_align_minirc.sh
./shellScripts/run_plotter_aligned_stim.sh