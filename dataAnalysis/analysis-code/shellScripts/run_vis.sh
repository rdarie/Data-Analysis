#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20200318

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignStim_20200318.stdout
#SBATCH -e ../../batch_logs/%j-alignStim_20200318.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo


module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python ./launchVis.py