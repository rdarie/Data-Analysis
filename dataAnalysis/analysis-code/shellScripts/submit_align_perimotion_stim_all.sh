#!/bin/bash

#
# shellScripts/run_analysis_maker_proprio.sh
#

sbatch shellScripts/run_align_perimotion_stim_lfp.sh
sbatch shellScripts/run_align_perimotion_stim_rig.sh
# sbatch shellScripts/run_align_perimotion_stim_fr.sh
# sbatch shellScripts/run_align_perimotion_stim_raster.sh