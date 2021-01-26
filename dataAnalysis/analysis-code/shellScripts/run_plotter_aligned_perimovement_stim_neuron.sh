#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_peri_neuron

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-plots_peri_neuron.stdout
#SBATCH -e ../../batch_logs/%j-%a-plots_peri_neuron.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_plotter_aligned_perimovement_stim_preamble.sh

python3 -u './plotAlignedNeurons.py' --exp=$EXP --unitQuery="all" --enableOverrides $BLOCKSELECTOR $ANALYSISFOLDER $UNITSELECTOR $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $PAGELIMITS $OTHERNEURONOPTS
