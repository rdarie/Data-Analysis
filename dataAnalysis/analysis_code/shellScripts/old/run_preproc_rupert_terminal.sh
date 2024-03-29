
# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J isi_preproc_one_shot

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-isi_preproc_one_shot.out
#SBATCH -e ../../batch_logs/%j-%a-isi_preproc_one_shot.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=6-9

#SBATCH --mail-type=ALL
#SBATCH --mail-user=radu_darie@brown.edu

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
# EXP="exp202012091400"

#
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version
#

LAZINESS="--lazy"
# LAZINESS=""

WINDOW="--window=XS"
# WINDOW="--window=XS"

# ANALYSISFOLDER="--analysisName=fullRes"
# ANALYSISFOLDER="--analysisName=hiRes"
ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=default"
# ANALYSISFOLDER="--analysisName=parameter_recovery"

CHANSELECTOR="--chanQuery=all"
# CHANSELECTOR="--chanQuery=isiemgraw"
# CHANSELECTOR="--chanQuery=isiemgoranalog"
# CHANSELECTOR="--chanQuery=isiemgoracc"
# CHANSELECTOR="--chanQuery=isispinal"
# CHANSELECTOR="--chanQuery=isispinaloremg"

# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinal"
# UNITSELECTOR="--unitQuery=isiemgoracc"
# UNITSELECTOR="--unitQuery=isiacc"
# UNITSELECTOR="--unitQuery=isispinaloremg"

SLURM_ARRAY_TASK_ID=5
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

#  #  preprocess
# python -u ./preprocNS5.py --exp=$EXP $BLOCKSELECTOR --ISIMinimal --transferISIStimLog
# python -u ./preprocDelsysHPF.py --exp=$EXP $BLOCKSELECTOR $CHANSELECTOR --verbose
# python -u ./preprocDelsysCSV.py --exp=$EXP $BLOCKSELECTOR $CHANSELECTOR --verbose
