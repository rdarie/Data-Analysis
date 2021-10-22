
import dataAnalysis.custom_transformers.tdr as tdr
import pdb

processSlurmTaskCount = 6

'''joblibBackendArgs = dict(
    backend='dask',
    daskComputeOpts=dict(
        scheduler='processes'
        ),
    # backend='loky',
    n_jobs=-1
    )'''
joblibBackendArgs = dict(
    backend='loky',
    n_jobs=-1
    )
addHistoryTerms = [
    # hto0
    {
        'nb': 6, 'logBasis': True,
        'dt': None,
        'historyLen': 240e-3,
        'b': 5e-2, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    # hto1
    {
        'nb': 6, 'logBasis': True,
        'dt': None,
        'historyLen': 240e-3,
        'b': 5e-2, 'useOrtho': False,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    # hto2
    {
        'nb': 6, 'logBasis': True,
        'dt': None,
        'historyLen': 60e-3,
        'b': 2e-2, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    # hto3
    {
        'nb': 6, 'logBasis': True,
        'dt': None,
        'historyLen': 60e-3,
        'b': 2e-2, 'useOrtho': False,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]

regressionColumnsToUse = ['velocity_abs', 'amplitude', 'RateInHz', 'electrode']
regressionColumnRenamer = {
    'velocity_abs': 'v', 'amplitude': 'a', 'RateInHz': 'r', 'electrode': 'e'
    }

def iWrap(x):
    return 'I({})'.format(x)

def elecWrap(x):
    return 'e:({})'.format(x)

rcb = tdr.patsyRaisedCosTransformer

def genRcbWrap(htoStr):
    def rcbWrap(x):
        return 'rcb({}, **{})'.format(x, htoStr)
    return rcbWrap

def genElecRcbWrap(htoStr):
    def elecRcbWrap(x):
        return 'e:rcb({}, **{})'.format(x, htoStr)
    return elecRcbWrap

def ddt(x):
    return x.diff().fillna(0)

def diffWrap(x):
    return('ddt({})'.format(x))

def abv(x):
    return x.abs()

def absWrap(x):
    return('abv({})'.format(x))

designFormulaTemplates = [
    '{v} + {a} + {r} - 1',
    ]

lOfDesignFormulas = []
#
masterExogFormulas = []
masterExogLookup = {}
# masterExogFormulas and masterExogLookup are used
# relate design formulas that are included in one another,
# to avoid re-computing terms from convolution
lOfHistTemplates = []
sourceTermDict = {}
sourceHistOptsDict = {}
designHistOptsDict = {}
templateHistOptsDict = {}
designIsLinear = {}
formulasShortHand = {}
#
lOfEndogAndExogTemplates = []
for lagSpecIdx in range(2):
    lagSpec = 'hto{}'.format(lagSpecIdx)
    wrapperFun = genRcbWrap(lagSpec)
    elecWrapperFun = genElecRcbWrap(lagSpec)
    laggedModels = {}
    for source in ['v']:
        laggedModels[source] = wrapperFun(source)
        sourceTermDict[wrapperFun(source)] = source
        sourceHistOptsDict[wrapperFun(source).replace(' ', '')] = addHistoryTerms[lagSpecIdx]
    for source in ['a', 'r', 'a*r', 'v*r', 'v*a', 'v*a*r']:
        laggedModels[source] = elecWrapperFun(source)
        sourceTermDict[elecWrapperFun(source)] = source
        sourceHistOptsDict[elecWrapperFun(source).replace(' ', '')] = addHistoryTerms[lagSpecIdx]
    #
    theseFormulas = [dft.format(**laggedModels) for dft in designFormulaTemplates]
    formulasShortHand.update({
        dft.format(**laggedModels): '({}, **{})'.format(dft, lagSpec)
        for dft in designFormulaTemplates})
    for fIdx in range(1):
        designIsLinear.update({
            theseFormulas[fIdx]: True})
    masterExogFormulas.append(theseFormulas[0])
    for tf in theseFormulas:
        masterExogLookup[tf] = theseFormulas[0]
    lOfDesignFormulas += theseFormulas
    designHistOptsDict.update({
        thisForm: addHistoryTerms[lagSpecIdx]
        for thisForm in theseFormulas})
    ########################
    lagSpecIdxHist = lagSpecIdx + 2
    lagSpecHist = 'hto{}'.format(lagSpecIdxHist)
    histTemplate = 'rcb({}, **{})'.format('{}', lagSpecHist)
    lOfHistTemplates.append(histTemplate)
    templateHistOptsDict[histTemplate] = addHistoryTerms[lagSpecIdxHist]
    for exogFormula in ['NULL'] + theseFormulas:
        for endogFormula in ['NULL', histTemplate]:
            if not ((exogFormula == 'NULL') and (endogFormula == 'NULL')):
                lOfEndogAndExogTemplates.append((exogFormula, endogFormula, endogFormula,))
#
lOfDesignFormulas.append('NULL')
lOfHistTemplates.append('NULL')
formulasShortHand['NULL'] = 'NULL'
designIsLinear['NULL'] = True
#
lOfEnsembleTemplates = [
    (hT, hT) for hT in lOfHistTemplates
    ]

lhsMasksOfInterest = {
    'plotPredictions': [1, 2],
    'varVsEnsemble': [2, 5]
    }
burnInPeriod = 250e-3

def getHistoryOpts(hTDict, iteratorOpts, rasterOpts):
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
    hTDict['dt'] = binInterval
    return hTDict

fullFormulaReadableLabels = {
    '({v} + {a} + {r} - 1, **hto1) + rcb(ensemble, **hto1) + rcb(self, **hto1)': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v + a + r)$',
    }

modelsTestReadable = {
    'ensemble_history': r'$\dot{x} = \mathbf{A}x$',
    '(v + a + r +) ensemble_history': '$\dot{x} = \mathbf{A}x + \mathbf{B}(\mathbf{v + a + r})$',
    '(v +) a + r': '$\dot{x} = \mathbf{A}x + \mathbf{B}(\mathbf{v} + a + r)$',
    'a + r': '$\dot{x} = \mathbf{A}x + \mathbf{B}(a + r)$',
    'v + (a + r)': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v \mathbf{+ a + r})$',
    'v': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v)$',
    'v + a + r (+ va + vr) + ar': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v + a + r \mathbf{+ v\dot a + v\dot r }+ a\dot r)$',
    'v + a + r + ar': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v + a + r + a\dot r)$',
    }