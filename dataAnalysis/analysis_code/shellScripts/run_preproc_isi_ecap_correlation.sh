#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=8:00:00

#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=127G

# Specify a job name:
#SBATCH -J isi_preproc_ecap_20201222

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-isi_preproc_ecap_20201222.out
#SBATCH -e ../../batch_logs/%j-%a-isi_preproc_ecap_20201222.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,3,4,5,6

# SBATCH --mail-type=ALL
# SBATCH --mail-user=radu_darie@brown.edu
# SBATCH --export=OUTDATED_IGNORE=1

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
# has blocks 3,5,6,7,8
# EXP="exp202012221300"
# has blocks 1,3,4,5,6
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

# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinal"
# UNITSELECTOR="--unitQuery=isispinal"
# UNITSELECTOR="--unitQuery=isiemgoracc"
# UNITSELECTOR="--unitQuery=isiacc"
# UNITSELECTOR="--unitQuery=isispinaloremg"

SLURM_ARRAY_TASK_ID=3
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

#  #  preprocess
python -u ./preprocNS5.py --exp=$EXP $BLOCKSELECTOR --ISIRaw --transferISIStimLog
CHANSELECTOR="--chanQuery=isiemgoranalog"
python -u ./preprocDelsysCSV.py --exp=$EXP $BLOCKSELECTOR $CHANSELECTOR --verbose
#  #  synchronize
CHANSELECTOR="--chanQuery=all"
python -u ./synchronizeDelsysToNSP.py $BLOCKSELECTOR --exp=$EXP $CHANSELECTOR --trigRate=2
#  #  downsample
python -u ./calcISIAnalysisNix.py --exp=$EXP $BLOCKSELECTOR $CHANSELECTOR $ANALYSISFOLDER --verbose

EVENTFOLDER="--eventSubfolder=loRes"
SIGNALFOLDER="--signalSubfolder=loRes"
OUTPUTBLOCKNAME="--outputBlockSuffix=emg"
CHANSELECTOR="--chanQuery=isiemg"
python -u ./calcAlignedAsigs.py --signalBlockSuffix="analyze" --eventBlockSuffix="analyze" --exp=$EXP $BLOCKSELECTOR $WINDOW $LAZINESS $ANALYSISFOLDER --eventName=stim $CHANSELECTOR $OUTPUTBLOCKNAME --verbose --alignFolderName=stim --amplitudeFieldName=nominalCurrent $EVENTFOLDER $SIGNALFOLDER
#
ANALYSISFOLDER="--analysisName=fullRes"
CHANSELECTOR="--chanQuery=isispinal"
OUTPUTBLOCKNAME="--outputBlockSuffix=lfp_raw"
# python -u ./calcAlignedAsigs.py $CHANSELECTOR $OUTPUTBLOCKNAME --eventBlockSuffix='analyze' $EVENTFOLDER --signalBlockPrefix='Block' --verbose --exp=$EXP --amplitudeFieldName=nominalCurrent $BLOCKSELECTOR $WINDOW $LAZINESS --eventName=stim --alignFolderName=stim $ANALYSISFOLDER --signalSubfolder=None
UNITSELECTOR="--unitQuery=isispinal"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_raw"
# python -u ./makeViewableBlockFromTriggered.py --plotting $INPUTBLOCKNAME $UNITSELECTOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS --alignFolderName=stim $OUTLIERMASK
