#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=2:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=16
#SBATCH --tasks=16
#SBATCH --tasks-per-node=1
#SBATCH --mem=56G

# Specify a job name:
#SBATCH -J constructor
#SBATCH --array=1

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
EXP="exp202009111100"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

module load mpi
# srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPreprocess --chan_start=0 --chan_stop=25 --arrayName=utah --sourceFile=processed --remakePrb --removeExistingCatalog
srun --mpi=pmi2 python3 -u ./tridesclousCCV.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPreprocess --chan_start=0 --chan_stop=16 --arrayName=nform --sourceFile=processed --remakePrb --removeExistingCatalog
