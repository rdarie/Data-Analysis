#!/bin/bash

sbatch shellScripts/run_align_motion_lfp.sh
sbatch shellScripts/run_align_motion_rig.sh
sbatch shellScripts/run_align_motion_fr.sh
sbatch shellScripts/run_align_motion_raster.sh