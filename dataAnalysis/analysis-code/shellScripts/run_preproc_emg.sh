#!/bin/bash
# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1,3,4

# Specify a job name:
#SBATCH -J emg_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-emg_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-emg_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
# EXP="exp201901271000"
# EXP="exp201812051000"
EXP="exp201901070700"
# SLURM_ARRAY_TASK_ID="2"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

python3 ./preprocOpenEphys.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --loadMat
python3 ./preprocINS.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID
python3 ./synchronizeOpenEphysToINS.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --plotting
# python3 ./preprocINSfromSIP.py --exp=$EXP
# python3 ./synchronizeOpenEphysToINSSIP.py --exp=$EXP
# python3 ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --makeTruncated
# python3 ./synchronizeOpenEphysToNSP.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID
# python3 ./calcBlockAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --chanQuery="all" --analysisName=shortGaussian
# python3 ./calcStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analysisName=shortGaussian
# python3 ./calcFR.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analysisName=shortGaussian
# python3 ./calcFRsqrt.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analysisName=shortGaussian
#
# python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters --analysisName="shortGaussian"
#
# python3 ./calcAlignedAsigs.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --chanQuery="oechorins" --blockName=RC --eventName=stimAlignTimes
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --inputBlockName="RC" --unitQuery="oechorins" --alignQuery="stimOnLowRate" --rowName= --colName="electrode" --colControl="control" --styleName= --hueName="amplitude"
#
# python3 ./calcAlignedAsigs.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --eventName=stimAlignTimes --chanQuery=fr --blockName=fr
# python3 ./calcAlignedRasters.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --eventName=stimAlignTimes --chanQuery=raster --blockName=raster
# python3 ./plotAlignedNeurons.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --alignQuery="stimOnLowRate" --rowName= --colName="electrode" --colControl="control" --styleName= --hueName="amplitude"
#
# python3 ./calcRecruitment.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --inputBlockName="RC" --unitQuery="oechorins" --alignQuery="stimOnLowRate" --rowName= --colName="electrode" --colControl="control" --styleName= --hueName="amplitude" --verbose
# python3 ./plotRecruitment.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --inputBlockName="RC" --unitQuery="oechorins" --alignQuery="stimOnLowRate" --rowName= --colName="electrode" --colControl="control" --styleName= --hueName="amplitude" --verbose
#
# python3 ./calcUnitCorrelationToAsig.py --exp=$EXP --processAll --window="RC" --lazy --analysisName="shortGaussian" --inputBlockName="fr" --secondaryBlockName="RC" --alignQuery="stimOnLowRate" --unitQuery="fr" --verbose
# python3 ./plotMatrixOfScalars.py --exp=$EXP --resultName="emgMaxCrossCorr" --processAll --window="RC" --analysisName="shortGaussian" --inputBlockName="fr" --secondaryBlockName="RC" --verbose
# python3 ./plotMatrixOfScalars.py --exp=$EXP --resultName="emgMaxCrossCorrLag" --processAll --window="RC" --analysisName="shortGaussian" --inputBlockName="fr" --secondaryBlockName="RC" --verbose