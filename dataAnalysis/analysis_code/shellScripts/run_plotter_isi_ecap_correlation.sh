#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J isi_preproc_17

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-isi_preproc_17.out
#SBATCH -e ../../batch_logs/%j-%a-isi_preproc_17.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
######SBATCH --array=6-9

#SBATCH --mail-type=ALL
#SBATCH --mail-user=radu_darie@brown.edu
#SBATCH --export=OUTDATED_IGNORE=1

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202003091200"
# EXP="exp202003131100"
# EXP="exp202003201200"
# EXP="exp202003191400"
# EXP="exp202004251400"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202003181300"
# EXP="exp202006171300"

# EXP="exp202007011300"
# has blocks 1,2,3,4

# EXP="exp202007021300"
# EXP="exp202007071300"
# EXP="exp202007081300"
# EXP="exp202008180700"
# EXP="exp202009031500"
# EXP="exp202009221300"
# EXP="exp202009231400"
# EXP="exp202010071400"
# EXP="exp202010081400"
EXP="exp202010151400"
# EXP="exp202010191100"
# EXP="exp202012171300"
# EXP="exp202012221300"

# #
module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version
#######

export OUTDATED_IGNORE=1
LAZINESS="--lazy"
# LAZINESS=""

WINDOW="--window=XXS"
# WINDOW="--window=XS"

SLURM_ARRAY_TASK_ID=1
# BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

INPUTBLOCKNAME="--inputBlockSuffix=emg"
ALIGNQUERY="--alignQuery=stimOnLessThan30Hz"
INPUTBLOCKNAME="--emgBlockSuffix=emg"
# python -u ./plotEcapEMGCorrelation.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR --alignQuery="stimOn"

ANALYSISFOLDER="--analysisName=fullRes"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_raw"
# python -u ./applyCARToLmFitResults.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR --alignQuery="stimOn"
# python -u ./plotLmFitPerformance.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR --alignQuery="stimOn"
#
ANALYSISFOLDER="--analysisNameLFP=fullRes --analysisNameEMG=loRes"
INPUTBLOCKNAME="--emgBlockSuffix=emg --lfpBlockSuffix=lfp_raw"
python -u ./plotEcapEMGCorrelationFromAuto.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR --alignQuery="stimOn" --showFigures
