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
#SBATCH -o ../batch_logs/%j-%a-spike_sort_peeler.stdout
#SBATCH -e ../batch_logs/%j-%a-spike_sort_peeler.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
EXP="exp201901261000"
TRIALIDX="4"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

# Constructor
# python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP  --batchPreprocess
# python3 ./tridesclousVisualize.py --blockIdx=$TRIALIDX --exp=$EXP  --constructor
# Transfer
python3 ./transferTDCTemplates.py --blockIdx=$TRIALIDX --exp=$EXP
# Peeler
# python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP --purgePeeler --batchPeel
# python3 ./tridesclousVisualize.py --blockIdx=$TRIALIDX --exp=$EXP  --peeler
#
# python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP --exportSpikesCSV
# python3 ./tridesclousCCV.py --blockIdx=$TRIALIDX --exp=$EXP --purgePeelerDiagnostics
# python3 './tridesclousCCV.py' --blockIdx=$TRIALIDX --makeStrictNeoBlock --exp=$EXP
# python3 './plotSpikeReport.py' --blockIdx=$TRIALIDX --nameSuffix=_final --exp=$EXP