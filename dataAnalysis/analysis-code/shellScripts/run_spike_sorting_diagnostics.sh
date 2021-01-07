#!/bin/bash
# 05: Assemble the spike nix file
# Request 24 hours of runtime:
#SBATCH --time=4:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J sorting_diagnostics

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-sorting-diagnostics.stdout
#SBATCH -e ../../batch_logs/%j-%a-sorting-diagnostics.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202012091200"
# EXP="exp202012101100"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012171200"
# EXP="exp202012181200"
EXP="exp202101051100"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1

########################################################################################################################################################################################################################

SOURCESELECTOR="--sourceFileSuffix=spike_preview"
# SOURCESELECTOR="--sourceFileSuffix=mean_subtracted"
# --sourceFileSuffix='spike_preview', --sourceFileSuffix='mean_subtracted'
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --exp=$EXP --chan_start=0 --chan_stop=96 $SOURCESELECTOR
python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah $SOURCESELECTOR
#
# python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --purgePeelerDiagnostics --exp=$EXP --chan_start=0 --chan_stop=96  --sourceFileSuffix='mean_subtracted'
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah  --sourceFileSuffix='mean_subtracted'

########################################################################################################################################################################################################################
