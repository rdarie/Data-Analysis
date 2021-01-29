#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=3:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J spike_sort_shell
#    SBATCH --array=2

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spike_sort_shell.out
#SBATCH -e ../../batch_logs/%j-%a-spike_sort_shell.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Run a command
source ./shellScripts/run_spike_sorting_preamble.sh

BLOCKIDX=4

SOURCESELECTOR="--sourceFileSuffix=spike_preview"
# SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"

CHAN_START=0
CHAN_STOP=96

############## init spike sorting
python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$BLOCKIDX --forSpikeSorting
python -u ./tridesclousCCV.py --blockIdx=$BLOCKIDX --exp=$EXP --arrayName=utah --sourceFileSuffix=spike_preview --removeExistingCatalog --initCatalogConstructor
##
# python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$BLOCKIDX --forSpikeSorting
# python -u ./tridesclousCCV.py --blockIdx=$BLOCKIDX --exp=$EXP --arrayName=nform --sourceFileSuffix=spike_preview --removeExistingCatalog --initCatalogConstructor

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
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeelerDiagnostics --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step 6: Export to NIX
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --makeStrictNeoBlock --exp=$EXP --chan_start=0 --chan_stop=96 $SOURCESELECTOR
# python -u ./plotSpikeReport.py --blockIdx=$BLOCKIDX --nameSuffix=_final --exp=$EXP --arrayName=utah $SOURCESELECTOR
