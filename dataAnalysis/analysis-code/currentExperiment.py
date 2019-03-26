from exp201901271000 import *
import os

#  should rename these to something more intuitive
#  paths relevant to individual trials
analysisDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['ns5FileName'] + '_analyze.nix')
trialBasePath = os.path.join(
    trialFilesFrom['utah']['folderPath'],
    trialFilesFrom['utah']['ns5FileName'] + '.nix')
insDataPath = os.path.join(
        trialFilesStim['ins']['folderPath'],
        trialFilesStim['ins']['experimentName'],
        trialFilesStim['ins']['ns5FileName'] + '_ins.nix'
    )

#  paths relevant to the entire experimental day
masterFeaturePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_MasterFeature.hdf')
estimatorPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_estimator.joblib')
experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_analyze.nix')
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_binarized.nix')
featurePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_features.hdf')
figureFolder = os.path.join(
    '..', 'figures'
    )

rasterOpts = {
    'binInterval': (1) * 1e-3, 'binWidth': (25) * 1e-3,
    'windowSize': (-.6, .6),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
