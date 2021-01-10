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
# EXP="exp201804271016"
# EXP="exp201805071032"
# EXP="exp201804240927"
# EXP="exp201805231100"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp201901261000"
# EXP="exp202008201100"
# EXP="exp202008261100"
# EXP="exp202008271200"
# EXP="exp202008281100"
# EXP="exp202008311100"
# EXP="exp202009021100"
# EXP="exp202009071200"
# EXP="exp202009101200"
# EXP="exp202009111100"
# EXP="exp202009211200"
# EXP="exp202009291300"
# EXP="exp202009301100"
# EXP="exp202010011100"
# EXP="exp202010271200"
# EXP="exp202011231200"
# EXP="exp202012171200"
# EXP="exp202012181200"
EXP="exp202101051100"
EXP="exp202101061100"

BLOCKIDX=3

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SOURCESELECTOR="--sourceFileSuffix=spike_preview"
# SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"

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

python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --chan_start=$CHAN_START --chan_stop=$CHAN_STOP $SOURCESELECTOR

# Step 6: Export to NIX
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --exp=$EXP --chan_start=0 --chan_stop=96 $SOURCESELECTOR
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah $SOURCESELECTOR