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
EXP="exp202009301100"
BLOCKIDX="1"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# Step 1: Constructor
python3 ./tridesclousCCV.py --blockIdx=$BLOCKIDX --exp=$EXP --chan_start=0 --chan_stop=32 --arrayName=utah --sourceFile=processed --remakePrb --removeExistingCatalog
# python3 ./tridesclousCCV.py --blockIdx=$BLOCKIDX --exp=$EXP --batchPreprocess --chan_start=0 --chan_stop=50 --arrayName=utah --sourceFile=processed --remakePrb --removeExistingCatalog

# Step 2: Validate the constructor
# python3 ./tridesclousVisualize.py --arrayName=utah --blockIdx=$BLOCKIDX --exp=$EXP  --constructor --chan_start=1 --chan_stop=2

# Step 3: Transfer the templates
# python3 ./transferTDCTemplates.py --arrayName=utah --exp=$EXP --chan_start=0 --chan_stop=50 --sourceFile=processed
# python3 ./transferTDCTemplates.py --arrayName=nform --exp=$EXP --chan_start=0 --chan_stop=32 --sourceFile=processed

# Step 4: Peeler
# python3 ./tridesclousCCV.py --blockIdx=$BLOCKIDX --exp=$EXP --purgePeeler --batchPeel
# python3 ./tridesclousVisualize.py --blockIdx=$BLOCKIDX --exp=$EXP  --peeler
# Step 5:
# python3 './tridesclousCCV.py' --blockIdx=$BLOCKIDX --purgePeelerDiagnostics --makeStrictNeoBlock --exp=$EXP
# python3 './plotSpikeReport.py' --blockIdx=$BLOCKIDX --nameSuffix=_final --exp=$EXP