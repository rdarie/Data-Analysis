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

# EXP="exp201901251000"
# has 1 minirc 2 motionsba
# EXP="exp201901261000"
# has 1-3 motion 4 minirc
# EXP="exp201901271000"
# has 1-4 motion 5 minirc
#
# EXP="exp201901311000"
# has 1-4 motion 5 minirc
# EXP="exp201902010900"
#  has 1-4 motion 5 minirc
# EXP="exp201902021100"
# has 3-5 motion 6 minirc; blocks 1 and 2 were bad;
# EXP="exp201902031100"
# has 1-4 motion 5 minirc;
# EXP="exp201902041100"
# has 1-4 motion 5 minirc;
# EXP="exp201902051100"
# has 1-4 motion
########
EXP="exp202101201100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101211100"
# has 1 minirc 2,3 motion+stim 4 motionOnly
EXP="exp202101221100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101251100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101271100"
# has 1 minirc 2 motion+stim 3 motionOnly
EXP="exp202101281100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202102021100"
# has 1 minirc 2 motion+stim 3 motionOnly

# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# radujobinfo
# Info about jobs for user 'rdarie' submitted since 2021-08-31T00:00:00
# Use option '-S' for a different date
#  or option '-j' for a specific Job ID.

# scratch to data
#
# full length files
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_chunkingInfo.json' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_synchFun.json' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Session*.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins.nix' --fromScratchToData --moveItems --filesIncluded

# everything ever
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.json' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.h5' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.pickle' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.joblib' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.csv' --fromScratchToData --moveItems --filesIncluded

###### data back to scratch
## required by qa_proprio
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*lfp_XL.nix' --fromDataToScratch --filesIncluded
## required to make epoch align times
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Session*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*analog_inputs*.*' --fromDataToScratch --filesIncluded
## required by test train splitter
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs*' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*rig_XL.nix' --fromDataToScratch --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp_CAR_scaled_XL.nix' --fromDataToScratch --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp_CAR_spectral_XL.nix' --fromDataToScratch --moveItems --filesIncluded
## required by plotting qa
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*lfp_XL.nix' --fromDataToScratch --filesIncluded

## bring back pre-synch files
# first, save existing files to processed (to make sure we don't reload old versions)
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_synchFun.json' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Session*.nix' --fromScratchToData --moveItems --filesIncluded
# then, bring everything back
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_synchFun.json' --fromDataToScratch --filesIncluded
python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*Session*.nix' --fromDataToScratch --filesIncluded

## bring back pre-analysis files
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah**_chunkingInfo.json --fromDataToScratch --filesIncluded

## bring back pre-align files
# first, save existing files to processed (to make sure we don't reload old versions)
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*analog_inputs.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_outliers.h5' --fromScratchToData --moveItems --filesIncluded
# then, bring everything back
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah005*analog_inputs.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block005*_analyze.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block005*_ins.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_outliers.h5' --fromDataToScratch --filesIncluded


## bring back aligned .nix files
# first, save existing files to processed (to make sure we don't reload old versions)
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_XL.nix' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_outliers.h5' --fromScratchToData --moveItems --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs.nix' --fromScratchToData --moveItems --filesIncluded
# then, bring everything back
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_XL.nix' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_outliers.h5' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_epochs.nix' --fromDataToScratch --filesIncluded

# bring back epoqued iterator dataframes
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_pa.h5' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_pa_*_meta.pickle' --fromDataToScratch --filesIncluded
# python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_pa_cvIterators.pickle' --fromDataToScratch --filesIncluded


# examine folder sizes
python './shuttleFilesToFromScratchV2.py' --exp=$EXP --printFolderSize
