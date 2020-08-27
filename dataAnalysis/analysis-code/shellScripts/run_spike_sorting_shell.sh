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
EXP="exp202008261100"
TRIALIDX="3"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# Step 1: Constructor
python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP  --batchPreprocess
# Step 2: Validate the constructor
# python3 ./tridesclousVisualize.py --blockIdx=$TRIALIDX --exp=$EXP  --constructor --chan_start=51 --chan_stop=64
# Step 3: Transfer
# python3 ./transferTDCTemplates.py --blockIdx=$TRIALIDX --exp=$EXP
# Step 4: Peeler
# python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP --purgePeeler --batchPeel
# python3 ./tridesclousVisualize.py --blockIdx=$TRIALIDX --exp=$EXP  --peeler
# Step 5:
# python3 './tridesclousCCV.py' --blockIdx=$TRIALIDX --purgePeelerDiagnostics --makeStrictNeoBlock --exp=$EXP
# python3 './plotSpikeReport.py' --blockIdx=$TRIALIDX --nameSuffix=_final --exp=$EXP