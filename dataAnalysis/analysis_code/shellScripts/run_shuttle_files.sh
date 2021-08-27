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

####
# EXP="exp202010151400"
# EXP="exp202012171300"
# EXP="exp202012221300"
# EXP="exp201901070700"
# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
#
# EXP="exp202101251100"
EXP="exp202101271100"
# EXP="exp202101281100"
#
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

python './shuttleFilesToFromScratchV2.py' --exp=$EXP --printFolderSize

# scratch to data
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_XL.nix' --fromScratchToData --moveItems --filesIncluded
###### data to scratch
## required by qa_proprio
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*lfp_XL.nix' --fromDataToScratch --filesIncluded
## required to make epoch align times
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*analog_inputs*.*' --fromDataToScratch --filesIncluded
## required by test train splitter
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs*' --fromDataToScratch --filesIncluded
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*rig_XL.nix' --fromDataToScratch --moveItems --filesIncluded
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp_CAR_XL.nix' --fromDataToScratch --moveItems --filesIncluded
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp_CAR_spectral_XL.nix' --fromDataToScratch --moveItems --filesIncluded

# global operations
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_lfp_*.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.*' --fromDataToScratch --filesIncluded
#
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins*' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp*.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins*' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.json' --fromScratchToData --moveItems --filesIncluded

#
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.parquet' --fromDataToScratch --foldersIncluded --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*loRes*' --fromDataToScratch --foldersIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*outlierTrials*' --fromDataToScratch --foldersIncluded

# everything to data
# python './shuttleFilesToFromScratch.py' --exp=$EXP --fromScratchToData --searchTerm='*' --moveItems --foldersIncluded --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --preprocFolderFiles --fromScratchToData --moveItems