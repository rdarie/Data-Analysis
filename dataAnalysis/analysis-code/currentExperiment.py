from exp201901271000 import *

rasterOpts = {
    'binInterval': (1) * 1e-3, 'binWidth': (26) * 1e-3,
    'windowSize': (-.6, .6),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'alignTo': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
