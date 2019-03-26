# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:07:44 2019

@author: Radu
"""

from importlib import reload

import matplotlib, os, pickle
matplotlib.use('TkAgg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # generate interactive output by default
experimentName = '201901271000-Proprio'
folderPath = './' + experimentName
trialIdx = 5

trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': folderPath,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': []}
    }

import dataAnalysis.ephyviewer.scripts as vis_scripts
vis_scripts.launch_standalone_ephyviewer()