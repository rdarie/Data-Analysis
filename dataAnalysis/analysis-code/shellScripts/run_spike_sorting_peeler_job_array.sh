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
# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
EXP="exp202011231200"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

BLOCKIDX=1
# SLURM_ARRAY_TASK_ID=0
let CHAN_START=SLURM_ARRAY_TASK_ID
# for nform, groups of 4 for utah, groups of 5
let CHAN_STOP=SLURM_ARRAY_TASK_ID+1

# python3 -u ./tridesclousCCV_jobArray.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='mean_subtracted'
# python3 -u ./tridesclousCCV_jobArray.py --arrayName=nform --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix=

python3 -u ./tridesclousCCV_jobArray.py --arrayName=utah --blockIdx=1 --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='mean_subtracted'
python3 -u ./tridesclousCCV_jobArray.py --arrayName=utah --blockIdx=2 --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='mean_subtracted'
python3 -u ./tridesclousCCV_jobArray.py --arrayName=utah --blockIdx=3 --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP --sourceFileSuffix='mean_subtracted'
