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
source ./shellScripts/run_spike_sorting_preamble.sh

BLOCKIDX=2

# SOURCESELECTOR="--sourceFileSuffix=spike_preview"
SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"

CHAN_START=0
CHAN_STOP=96
# Step 1: Init catalog
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --removeExistingCatalog --initCatalogConstructor $SOURCESELECTOR

# Step 2: Constructor
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPreprocess --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step 3: Validate the constructor
# python -u ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --constructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step X: Optional Remake the catalog
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchCleanConstructor --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step 4: Transfer the templates
# python ./transferTDCTemplates.py --arrayName=utah --exp=$EXP --chan_start=0 --chan_stop=96
# python ./transferTDCTemplates.py --arrayName=nform --exp=$EXP --chan_start=0 --chan_stop=65

# Step 5: Peeler (optional --purgePeeler)
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --batchPeel --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
# optional: visualize the peeler
# python -u ./tridesclousVisualize.py --blockIdx=$BLOCKIDX --exp=$EXP --peeler --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# If need to redo peeling
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeelerDiagnostics --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step 6: Export to NIX
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --exp=$EXP --chan_start=0 --chan_stop=96 $SOURCESELECTOR
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah $SOURCESELECTOR