#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J isiProc_20200309

# Specify an output file
#SBATCH -o ../batch_logs/%j-isiProc_20200309.stdout
#SBATCH -e ../batch_logs/%j-isiProc_20200309.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="expRippleSaline"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202001231400"
#
# EXP="exp202003091200"
# EXP="exp202003131100"
EXP="exp202003201200"

LAZINESS="--lazy"
WINDOW="--window=extraShort"
# TRIALSELECTOR="--blockIdx=2"
# TRIALSELECTOR="--processAll"
UNITSELECTOR="--selector=_minfrmaxcorrminamp"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

SLURM_ARRAY_TASK_ID="1"
python launchVis.py
# python ./quickPlotRipple.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID
# python ./saveImpedances.py --exp=$EXP --processAll --ripple --plotting
#
# python ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --ISI
#
# python ./calcISIAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --chanQuery="all"
# python3 ./calcAlignedAsigs.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $WINDOW $LAZINESS --eventName=stimAlignTimes --chanQuery="all" --blockName="lfp"  --alignFolderName=stim
# python3 ./plotAlignedAsigs.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $WINDOW --inputBlockName="lfp" --unitQuery="all" --alignQuery="stimOn" --rowName= --rowControl= --colControl= --hueName="amplitudeCat" --alignFolderName=stim --enableOverrides
