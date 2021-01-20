#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plotsStim

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

./shellScripts/run_plotter_aligned_stim_neurons.sh
./shellScripts/run_plotter_aligned_stim_lfp.sh
./shellScripts/run_plotter_aligned_stim_rig.sh