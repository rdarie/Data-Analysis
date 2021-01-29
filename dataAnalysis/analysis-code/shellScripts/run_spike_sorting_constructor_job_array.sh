#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=12:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J utah_constructor
#SBATCH --array=0-95:1

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-utah_constructor.out
#SBATCH -e ../../batch_logs/%j-%a-utah_constructor.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command
source ./shellScripts/run_spike_sorting_preamble.sh

BLOCKIDX=3
# SLURM_ARRAY_TASK_ID=0
let CHAN_START=SLURM_ARRAY_TASK_ID
# for nform, groups of 4 for utah, groups of 5
let CHAN_STOP=SLURM_ARRAY_TASK_ID+1

SOURCESELECTOR="--sourceFileSuffix=spike_preview"
# SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"

python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPreprocess --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP  --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
