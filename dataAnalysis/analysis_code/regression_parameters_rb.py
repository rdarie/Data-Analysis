
import dataAnalysis.custom_transformers.tdr as tdr
import pdb

processSlurmTaskCountPLS = 3
processSlurmTaskCount = 48
processSlurmTaskCountTransferFunctions = 48
joblibBackendArgs = dict(
    backend='loky',
    n_jobs=-1
    )
addHistoryTerms = [
    # hto0
    {
        'nb': 5, 'logBasis': True,
        'dt': None,
        'historyLen': 250e-3,
        'b': 2e-2, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]

regressionColumnsToUse = [
    # 'velocity_abs',
    'velocity_x', 'velocity_y',
    # 'velocity_x_abs', 'velocity_y_abs',
    'position_x', 'position_y',
    'amplitude', 'electrode', 'RateInHz',
    ]
regressionColumnRenamer = {
    'velocity_abs': 'v', 'amplitude': 'a',
    'electrode': 'e', 'RateInHz': 'r',
    'velocity_x': 'vx', 'velocity_y': 'vy',
    'position_x': 'px', 'position_y': 'py',
    # 'velocity_x_abs': 'vxa', 'velocity_y_abs': 'vya',
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
    '{vx} + {vy} + {px} + {py} + {a} + {r} - 1',
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
#
for lagSpecIdx in range(len(addHistoryTerms)):
    lagSpec = 'hto{}'.format(lagSpecIdx)
    wrapperFun = genRcbWrap(lagSpec)
    elecWrapperFun = genElecRcbWrap(lagSpec)
    laggedModels = {}
    for source in ['vx', 'vy', 'px', 'py', 'vxa', 'vya']:
        laggedModels[source] = wrapperFun(source)
        sourceTermDict[wrapperFun(source)] = source
        sourceHistOptsDict[wrapperFun(source).replace(' ', '')] = addHistoryTerms[lagSpecIdx]
    for source in [
        'a', 'vx*a', 'vy*a', 'vxa*a', 'vya*a',
        'r', 'vx*r', 'vy*r', 'vxa*r', 'vya*r',
        'a', 'px*a', 'py*a', 'pxa*a', 'pya*a',
        'r', 'px*r', 'py*r', 'pxa*r', 'pya*r',
        'a*r', 'vx*a*r', 'vy*a*r', 'vxa*a*r', 'vya*a*r',
        'a*r', 'pvx*a*r', 'py*a*r', 'pxa*a*r', 'pya*a*r',]:
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
    histTemplate = 'rcb({}, **{})'.format('{}', lagSpec)
    lOfHistTemplates.append(histTemplate)
    designHistOptsDict.update({
        thisForm: addHistoryTerms[lagSpecIdx]
        for thisForm in theseFormulas})
    templateHistOptsDict[histTemplate] = addHistoryTerms[lagSpecIdx]
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
#
lhsMasksOfInterest = {
    'plotPredictions': [1, 2],
    'varVsEnsemble': [2]
    }
# 
burnInPeriod = 150e-3
#
def getHistoryOpts(hTDict, iteratorOpts, rasterOpts):
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
    hTDict['dt'] = binInterval
    return hTDict

fullFormulaReadableLabels = {
    '({v} + {a} - 1, **hto0) + rcb(ensemble, **hto0) + rcb(self, **hto0)': '$\dot{x} = \mathbf{A}x + \mathbf{B}(v + a)$',
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