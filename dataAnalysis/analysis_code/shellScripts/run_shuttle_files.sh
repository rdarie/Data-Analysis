#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J ins_preproc

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_preproc.out
#SBATCH -e ../../batch_logs/%j-%a-ins_preproc.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

#
# EXP="exp202010151400"
# EXP="exp202012171300"
# EXP="exp202012221300"
# EXP="exp201901070700"
# EXP="exp202101141100"
# EXP="exp202101191100"
EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"

# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# global operations
# python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderFiles --fromScratchToData --moveItems
# python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderFiles --preprocFolderSubfolders --fromScratchToData
# python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderFiles --fromDataToScratch
# python './shuttleFilesToFromScratch.py' --exp=$EXP --fileSearchTerm='*.parquet' --fromDataToScratch
# python './shuttleFilesToFromScratch.py' --exp=$EXP --fileSearchTerm='*_emg_XXS*' --fromDataToScratch
# python './shuttleFilesToFromScratch.py' --exp=$EXP --fileSearchTerm='*epochs*' --fromDataToScratch

# everything to data
python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderSubfolders --fromScratchToData --moveItems
python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderFiles --fromScratchToData --moveItems