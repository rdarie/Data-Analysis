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

# EXP="exp201901070700"
# EXP="exp201901211000"
# EXP="exp201901201200"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202008261100"
# EXP="exp202008271200"
# EXP="exp202008281100"
# EXP="exp202008311100"
# EXP="exp202009021100"
# EXP="exp202009071200"
# EXP="exp202009101200"
# EXP="exp202009111100"
# EXP="exp202009211200"
# EXP="exp202009301100"
# EXP="exp202010011100"
EXP="exp202010271200"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=3
# python -u ./tridesclousCCV.py --arrayName=nform --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --purgePeelerDiagnostics --exp=$EXP --chan_start=0 --chan_stop=32 --sourceFile=processed
# python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=nform
# SLURM_ARRAY_TASK_ID=1
python -u ./tridesclousCCV.py --arrayName=utah --blockIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --purgePeelerDiagnostics --exp=$EXP --chan_start=0 --chan_stop=50 --sourceFile=processed
python -u ./plotSpikeReport.py --blockIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP --arrayName=utah