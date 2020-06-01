#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=3:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J spike_sort_vis_parallel
#SBATCH --array=0,4,8,12,16,20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spike_sort_vis_parallel.stdout
#SBATCH -e ../../batch_logs/%j-%a-spike_sort_vis_parallel.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command
# EXP="exp201804271016"
# EXP="exp201805071032"
# EXP="exp201804240927"
EXP="exp201805231100"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp201901261000"
TRIALIDX="1"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# Step 2: Validate the constructor
let CHAN_START=SLURM_ARRAY_TASK_ID
let CHAN_STOP=SLURM_ARRAY_TASK_ID+4
python3 -u ./tridesclousVisualize.py --blockIdx=$TRIALIDX --exp=$EXP  --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP
