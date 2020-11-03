#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=4:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J constructor
###########SBATCH --array=0-28:4
##### SBATCH --array=0-49

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-constructor.stdout
#SBATCH -e ../../batch_logs/%j-%a-constructor.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command
# EXP="exp201805071032"
# EXP="exp201804271016"
# EXP="exp201804240927"
# EXP="exp201805231100"
# EXP="exp201901070700"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202008201100"
# EXP="exp202008261100"
# EXP="exp202008271200"
# EXP="exp202008281100"
# EXP="exp202008311100"
# EXP="exp202009021100"
# EXP="exp202009071200"
# EXP="exp202009101200"
# EXP="exp202009111100"
# EXP="exp202009211200"
# EXP="exp202009291300"
EXP="exp202009301100"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

BLOCKIDX=2
# SLURM_ARRAY_TASK_ID=0

# python testTensorFlowGPU.py
python3 ./tridesclousCCV_jobArray.py --blockIdx=$BLOCKIDX --exp=$EXP --chan_start=0 --chan_stop=50 --batchPreprocess --arrayName=utah --sourceFile=processed --remakePrb