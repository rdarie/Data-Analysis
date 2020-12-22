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
#SBATCH -o ../../batch_logs/%j-%a-utah_constructor.stdout
#SBATCH -e ../../batch_logs/%j-%a-utah_constructor.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011271100"
# EXP="exp202012071100"
# EXP="exp202012081200"
# EXP="exp202012091200"
# EXP="exp202012101100"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012151200"
# EXP="exp202012161200"
EXP="exp202012171200"
# EXP="exp202012181200"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

BLOCKIDX=2
# SLURM_ARRAY_TASK_ID=0
let CHAN_START=SLURM_ARRAY_TASK_ID
# for nform, groups of 4 for utah, groups of 5
let CHAN_STOP=SLURM_ARRAY_TASK_ID+1

SOURCESELECTOR="--sourceFileSuffix=spike_preview"
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPreprocess --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP  --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
