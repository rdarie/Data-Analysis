#!/bin/bash

sbatch shellScripts/run_ols_aligned_motion_lfp_trial_averaged_prep_spectral_fa.sh
sbatch shellScripts/run_ols_aligned_motion_lfp_trial_averaged_prep_spectral_pca.sh
sbatch shellScripts/run_ols_aligned_motion_lfp_trial_averaged_prep_time_domain_fa.sh
sbatch shellScripts/run_ols_aligned_motion_lfp_trial_averaged_prep_time_domain_pca.sh