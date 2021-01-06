#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=3:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=32
#SBATCH --tasks=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J spike_sort_peeler
#SBATCH --array=2

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spike_sort_peeler.stdout
#SBATCH -e ../../batch_logs/%j-%a-spike_sort_peeler.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012171200"
EXP="exp202101051100"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# Step 3: Transfer the templates
python ./transferTDCTemplates.py --arrayName=utah --exp=$EXP --chan_start=0 --chan_stop=96
