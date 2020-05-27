#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=12:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=26
#SBATCH --tasks=26
#SBATCH --tasks-per-node=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J spike_sort_constructor
#SBATCH --array=1

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spike_sort_constructor.stdout
#SBATCH -e ../../batch_logs/%j-%a-spike_sort_constructor.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command
# EXP="exp201901070700"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
EXP="exp201901271000"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

module load mpi
srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPreprocess
