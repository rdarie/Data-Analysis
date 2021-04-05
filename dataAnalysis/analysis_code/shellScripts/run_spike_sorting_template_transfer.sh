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
#SBATCH --array=1,2,3

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spike_sort_peeler.out
#SBATCH -e ../../batch_logs/%j-%a-spike_sort_peeler.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
source ./shellScripts/run_spike_sorting_preamble.sh

# Step 3: Transfer the templates
python ./transferTDCTemplates.py --arrayName=utah --exp=$EXP --chan_start=0 --chan_stop=96
