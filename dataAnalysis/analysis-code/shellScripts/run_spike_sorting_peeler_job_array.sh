#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=12:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J peeler_0126_4

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-peeler_0126_4.stdout
#SBATCH -e ../../batch_logs/%j-%a-peeler_0126_4.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=0-95:1

# Run a command
source ./shellScripts/run_spike_sorting_preamble.sh

# BLOCKIDX=1
# SLURM_ARRAY_TASK_ID=0
let CHAN_START=SLURM_ARRAY_TASK_ID
# for nform, groups of 4 for utah, groups of 5
let CHAN_STOP=SLURM_ARRAY_TASK_ID+1

# SOURCESELECTOR="--sourceFileSuffix=spike_preview"
SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
# python -u ./tridesclousCCV.py --arrayName=nform --blockIdx=$BLOCKIDX --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=1 --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=2 --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
