#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=12:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=25
#SBATCH --tasks=25
#SBATCH --tasks-per-node=1
#SBATCH --mem=56G

# Specify a job name:
#SBATCH -J peeler

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-peeler.stdout
#SBATCH -e ../../batch_logs/%j-%a-peeler.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# Run a command
# EXP="exp201804271016"
# EXP="exp201805231100"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202008261100"
# EXP="exp202008271200"
# EXP="exp202008281100"
# EXP="exp202008311100"
# EXP="exp202009021100"
# EXP="exp202009071200"
# EXP="exp202009101200"
# EXP="exp202009111100"
EXP="exp202009211200"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

module load mpi
srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --attemptMPI --purgePeeler --batchPeel --chan_start=0 --chan_stop=25 --sourceFile=processed
# srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --arrayName=nform --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --attemptMPI --purgePeeler --batchPeel --chan_start=0 --chan_stop=16 --sourceFile=processed
