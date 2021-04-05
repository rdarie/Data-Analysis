#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=3:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J sort_vis

#SBATCH --array=0-95:1

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-sort_vis.out
#SBATCH -e ../../batch_logs/%j-%a-sort_vis.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
EXP="exp202011231200"
BLOCKIDX="1"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# Step 2: Validate the constructor
let CHAN_START=SLURM_ARRAY_TASK_ID

# for nform, groups of 4
# for utah, groups of 5
let CHAN_STOP=SLURM_ARRAY_TASK_ID+1

python -u ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP  --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='spike_preview'
# python -u ./tridesclousVisualize.py --arrayName=nform --blockIdx=$BLOCKIDX --exp=$EXP  --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='spike_preview'

# python -u ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --peeler --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='spike_preview'
# python -u ./tridesclousVisualize.py --arrayName=nform --blockIdx=$BLOCKIDX --exp=$EXP  --peeler --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='spike_preview'