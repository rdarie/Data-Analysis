#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J analysis_calc_2021_01_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-analysis_calc_2021_01_20.out
#SBATCH -e ../../batch_logs/%j-%a-analysis_calc_2021_01_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101061100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"

LAZINESS="--lazy"
#
# ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=hiRes"
# ANALYSISFOLDER="--analysisName=normalizedByImpedance"
ANALYSISFOLDER="--analysisName=default"
#
SPIKEBLOCKSUFFIX="--spikeFileSuffix=mean_subtracted"
#
# SPIKESOURCE="--spikeSource=tdc"
SPIKESOURCE=""
#
RIGSUFFIX="--rigFileSuffix=analog_inputs"
# BLOCKSUFFIX=""
#
BLOCKPREFIX="--sourceFilePrefix=utah"
#
module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg fftw scons
module load anaconda/2020.02
module load mpi
# module load opengl
module load qt/5.10.1
module load zlib/1.2.11
module unload python

. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=2
echo --blockIdx=$SLURM_ARRAY_TASK_ID
# python -u ./synchronizeSIMItoNSP_stimBased.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --showFigures --inputBlockSuffix=analog_inputs  --inputBlockPrefix=utah --plotting --forceRecalc
# python -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
# python -u ./calcProprioAnalysisNix.py --hasKinematics --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
#
python -u ./calcMotionAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID  $ANALYSISFOLDER --plotParamHistograms $LAZINESS
# python -u ./calcRefinedStimAlignTimes.py --removeLabels='stimOff' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS
python -u ./calcRefinedStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS
python -u ./calcMotionStimAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $LAZINESS --plotParamHistograms
#
# python -u ./calcFR.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER
# python -u ./calcFRsqrt.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID

# RC
# python -u ./calcProprioAnalysisNix.py --hasEMG --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER $SPIKESOURCE $SPIKEBLOCKSUFFIX $BLOCKPREFIX $RIGSUFFIX --chanQuery="all" --verbose --lazy
# python -u ./calcRefinedStimAlignTimes.py --disableRefinement --removeLabels='stimOff' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $ANALYSISFOLDER --inputNSPBlockSuffix=analog_inputs --plotParamHistograms $LAZINESS
