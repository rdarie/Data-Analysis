#!/bin/bash
# 05: Assemble the spike nix file
# Request 24 hours of runtime:
#SBATCH --time=4:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J 2021_01_20_sorting_diagnostics

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-2021_01_20_sorting-diagnostics.stdout
#SBATCH -e ../../batch_logs/%j-%a-2021_01_20_sorting-diagnostics.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

source ./shellScripts/run_spike_sorting_preamble.sh

# SLURM_ARRAY_TASK_ID=1

########################################################################################################################################################################################################################

# SOURCESELECTOR="--sourceFileSuffix=spike_preview"
SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"
# --sourceFileSuffix='spike_preview', --sourceFileSuffix='mean_subtracted'
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --exp=$EXP --chan_start=0 --chan_stop=96 $SOURCESELECTOR
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah $SOURCESELECTOR
# #
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --purgePeelerDiagnostics --exp=$EXP --chan_start=0 --chan_stop=96  --sourceFileSuffix='mean_subtracted'
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah  --sourceFileSuffix='mean_subtracted'

########################################################################################################################################################################################################################
