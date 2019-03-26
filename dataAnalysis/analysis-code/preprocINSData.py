
import dataAnalysis.preproc.mdt as preprocINS

from currentExperiment import *
import dataAnalysis.ephyviewer.scripts as vis_scripts

#  vis_scripts.launch_standalone_ephyviewer()
insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'], plottingFigures=False)