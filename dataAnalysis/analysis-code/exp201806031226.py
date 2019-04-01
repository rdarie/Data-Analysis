#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

trialIdx = 1
miniRCTrial = False
#  plottingFigures = False
plottingFigures = False
plotBlocking = True

experimentName = '201806031226-Proprio'

#  remote paths
remoteBasePath = '..'
#  remoteBasePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings'
folderPath = os.path.join(remoteBasePath, experimentName)

#  should rename to something more intuitive
eventInfo = {
    'inputIDs': {
        'A+': 1,
        'B+': 2,
        'Z+': 3,
        'A-': 5,
        'B-': 4,
        'Z-': 6,
        'rightBut': 11,
        'leftBut': 12,
        'rightLED': 9,
        'leftLED': 10,
        'simiTrigs': 8,
        }
    }

for key, value in eventInfo['inputIDs'].items():
    eventInfo['inputIDs'][key] = 'ainp{}'.format(value)

trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': folderPath,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': []
        }
    }
trialFilesFrom['utah'].update(dict(eventInfo=eventInfo))

nspPrbPath = os.path.join('.', 'nsp_map.prb')
triFolder = os.path.join(
    trialFilesFrom['utah']['folderPath'],
    'tdc_' + trialFilesFrom['utah']['ns5FileName'])

# make .prb file for spike sorting
#  {'xcoords': 2, 'ycoords': 2}
nspCmpPath = os.path.join('.', 'nsp_map.cmp')
cmpDF = tdch.cmpToDF(nspCmpPath)
tdch.cmpDFToPrb(
    cmpDF, filePath=nspPrbPath,
    names=['elec', 'nform'],
    groupIn={'xcoords': 1, 'ycoords': 1}, appendDummy=16)
