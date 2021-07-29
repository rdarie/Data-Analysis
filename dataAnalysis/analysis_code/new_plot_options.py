
import dataAnalysis.custom_transformers.tdr as tdr
import pdb

statsTestOpts = dict(
    testStride=25e-3,
    testWidth=25e-3,
    tStart=0,
    tStop=None,
    pThresh=5e-2,
    correctMultiple=False
    )
relplotKWArgs = dict(
    errorbar='se',
    # ci=95, n_boot=1000,
    estimator='mean',
    # estimator=None, units='t',
    palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1",
    # facet_kws={'sharey': True},
    height=1.5, aspect=2, kind='line', rasterized=True)
vLineOpts = {'color': 'm', 'alpha': 0.5}
asigPlotShadingOpts = {
    'facecolor': vLineOpts['color'],
    'alpha': 0.1, 'zorder': -100}
asigSigStarOpts = {
    'c': vLineOpts['color'],
    # 'linestyle': 'None',
    's': 50,
    'marker': '*'
    }
nrnRelplotKWArgs = dict(
    palette="ch:1.6,-.3,dark=.1,light=0.7,reverse=1",
    func1_kws={
        'marker': 'd',
        'edgecolor': None,
        'edgecolors': 'face',
        'alpha': .3, 'rasterized': True},
    func2_kws={'ci': 'sem'},
    facet1_kws={'sharey': False},
    facet2_kws={'sharey': True},
    height=4, aspect=2,
    kind1='scatter', kind2='line')
nrnVLineOpts = {'color': 'y'}
nrnBlockShadingOpts = {
    'facecolor': nrnVLineOpts['color'],
    'alpha': 0.3, 'zorder': -100}
nrnSigStarOpts = {
    'c': nrnVLineOpts['color'],
    # 'edgecolor': None,
    'edgecolors': 'face',
    # 'linestyle': 'None',
    's': 20,
    'marker': '*'}
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 22),
    'removeOutliers': (0.01, 0.975)}