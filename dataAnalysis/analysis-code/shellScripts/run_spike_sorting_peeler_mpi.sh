#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=24:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=4
#SBATCH --tasks=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J peeler

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-peeler.stdout
#SBATCH -e ../../batch_logs/%j-%a-peeler.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

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
EXP="exp202009101200"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

module load mpi
# srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --attemptMPI --purgePeeler --batchPeel --chan_start=0 --chan_stop=16
srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --attemptMPI --purgePeeler --batchPeel --chan_start=0 --chan_stop=16 --arrayName=nform
