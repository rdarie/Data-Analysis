#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=72:00:00

#SBATCH --ntasks=127
#SBATCH --ntasks-per-core=127
#SBATCH --mem-per-cpu=16G

# Specify a job name:
#SBATCH -J isi_preproc_lfp_17

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-isi_preproc_lfp_17.out
#SBATCH -e ../../batch_logs/%j-%a-isi_preproc_lfp_17.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#  UNUSED SBATCH --array=2-4

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
# EXP="exp202010151400"
# EXP="exp202010191100"
EXP="exp202012171300"
# EXP="exp202012221300"

# 
module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version
#

export OUTDATED_IGNORE=1
LAZINESS="--lazy"
# LAZINESS=""

WINDOW="--window=XXS"
# WINDOW="--window=XS"

ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=hiRes"
# ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=default"
# ANALYSISFOLDER="--analysisName=parameter_recovery"


# CHANSELECTOR="--chanQuery=isiemgraw"
# CHANSELECTOR="--chanQuery=isiemgoranalog"
# CHANSELECTOR="--chanQuery=isiemgoracc"
# CHANSELECTOR="--chanQuery=isispinal"
# CHANSELECTOR="--chanQuery=isispinaloremg"


SLURM_ARRAY_TASK_ID=3
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
INPUTBLOCKNAME="--inputBlockSuffix=emg"
ANALYSISFOLDER="--analysisName=loRes"
# python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $INPUTBLOCKNAME --alignFolderName=stim

INPUTBLOCKNAME="--inputBlockSuffix=emg"
UNITSELECTOR="--unitQuery=isiemgenv"
ALIGNQUERY="--alignQuery=stimOn"
# python -u ./calcTrialOutliers.py --exp=$EXP --alignFolderName=stim $INPUTBLOCKNAME $BLOCKSELECTOR $ANALYSISFOLDER $UNITSELECTOR $WINDOW $ALIGNQUERY --verbose --plotting --saveResults

# OUTLIERMASK=""
OUTLIERMASK="--maskOutlierBlocks"

ALIGNQUERY="--alignQuery=stimOnLessThan30Hz"
UNITSELECTOR="--unitQuery=isiemgraw"
# python -u ./plotAlignedAsigs.py --winStart=50 --winStop=80 --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --rowName="electrode" --rowControl= --colName="RateInHz" --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides
# python -u ./plotRippleStimSpikeReport.py --winStart=20 --winStop=80 --exp=$EXP $BLOCKSELECTOR $WINDOW $UNITSELECTOR $ANALYSISFOLDER $ALIGNQUERY --alignFolderName=stim $INPUTBLOCKNAME --groupPagesBy="electrode, RateInHz" $OUTLIERMASK

UNITSELECTOR="--unitQuery=isiemgraw"
ALIGNQUERY="--alignQuery=stimOn"
python -u ./calcRecruitment.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY $OUTLIERMASK
python -u ./plotRecruitment.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --showFigures

ANALYSISFOLDER="--analysisName=fullRes"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_raw"
# python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $INPUTBLOCKNAME --alignFolderName=stim

ALIGNQUERY="--alignQuery=stimOnLessThan30Hz"
UNITSELECTOR="--unitQuery=isispinal"
# python -u ./plotAlignedAsigs.py --winStart=1 --winStop=4 --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --rowName="electrode" --rowControl= --colName="RateInHz" --colControl= --hueName="nominalCurrent" --alignFolderName=stim --enableOverrides $OUTLIERMASK
# python -u ./plotRippleStimSpikeReport.py --winStart=1 --winStop=4 --exp=$EXP $BLOCKSELECTOR $WINDOW $UNITSELECTOR $ANALYSISFOLDER $ALIGNQUERY --alignFolderName=stim $INPUTBLOCKNAME --groupPagesBy="electrode, RateInHz" $OUTLIERMASK
#
python -u ./calcLFPLMFitModel.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY --plotting --debugging --smallDataset --interactive