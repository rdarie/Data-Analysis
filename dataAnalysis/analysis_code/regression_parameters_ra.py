
import dataAnalysis.custom_transformers.tdr as tdr
import pdb

validTargetLhsMaskIdx = {
    'ols_select_scaled': [0, 1, 2, 3,],
    'ols2_select2_scaled': [4, 5, 6],
    'ols_select_spectral_scaled': [0, 1, 2, 3,],
    'ols2_select2_spectral_scaled': [4, 5, 6]
}

processSlurmTaskCountPLS = 3
processSlurmTaskCount = 23
processSlurmTaskCountTransferFunctions = 23
joblibBackendArgs = dict(
    backend='loky',
    n_jobs=-1
    )
addEndogHistoryTerms = [
    # enhto0
    {
        'nb': 3, 'logBasis': True,
        'dt': None,
        'historyLen': 520e-3,
        'timeDelay': 80e-3,
        'b': 50e-3, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]
addExogHistoryTerms = [
    # exhto0
    {
        'nb': 3, 'logBasis': True,
        'dt': None,
        'historyLen': 600e-3,
        'timeDelay': 0.,
        'b': 50e-3, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]

regressionColumnsToUse = [
    # 'velocity_abs',
    'velocity_x', 'velocity_y',
    # 'velocity',
    # 'velocity_x_abs', 'velocity_y_abs',
    # 'position_x', 'position_y',
    # 'position',
    'amplitude', 'electrode', 'RateInHz',
    ]
regressionColumnRenamer = {
    'amplitude': 'a',
    'electrode': 'e', 'RateInHz': 'r',
    'position': 'p', 'velocity': 'v',
    'velocity_x': 'vx', 'velocity_y': 'vy',
    'position_x': 'px', 'position_y': 'py',
    'velocity_abs': 'absv',
    'velocity_x_abs': 'absvx', 'velocity_y_abs': 'absvy',
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
    '{vx} + {vy} + {a} + {r} - 1',
    '{vx} + {vy} - 1',
    '{a} + {r} - 1',
    ]

#
# masterExogFormulas and masterExogLookup are used
# relate design formulas that are included in one another,
# to avoid re-computing terms from convolution
masterExogFormulas = []
masterExogLookup = {}
#
lOfDesignFormulas = []
sourceTermDict = {}
sourceHistOptsDict = {}
designHistOptsDict = {}
designIsLinear = {}
formulasShortHand = {}
#
#
for lagSpecIdx in range(len(addExogHistoryTerms)):
    lagSpec = 'exhto{}'.format(lagSpecIdx)
    wrapperFun = genRcbWrap(lagSpec)
    elecWrapperFun = genElecRcbWrap(lagSpec)
    laggedModels = {}
    for source in ['vx', 'vy', 'v', 'absvx', 'absvy', 'p']:
        laggedModels[source] = wrapperFun(source)
        sourceTermDict[wrapperFun(source)] = source
        sourceHistOptsDict[wrapperFun(source).replace(' ', '')] = addExogHistoryTerms[lagSpecIdx]
    for source in [
        'a', 'vx*a', 'vy*a',
        'r', 'vx*r', 'vy*r',
        'a*r', 'vx*a*r', 'vy*a*r',]:
        laggedModels[source] = elecWrapperFun(source)
        sourceTermDict[elecWrapperFun(source)] = source
        sourceHistOptsDict[elecWrapperFun(source).replace(' ', '')] = addExogHistoryTerms[lagSpecIdx]
    #
    theseFormulas = [dft.format(**laggedModels) for dft in designFormulaTemplates]
    formulasShortHand.update({
        dft.format(**laggedModels): '({}, **{})'.format(dft, lagSpec)
        for dft in designFormulaTemplates})
    for fIdx in range(len(theseFormulas)):
        designIsLinear.update({
            theseFormulas[fIdx]: True})
    masterExogFormulas.append(theseFormulas[0])
    for tf in theseFormulas:
        masterExogLookup[tf] = theseFormulas[0]
    lOfDesignFormulas += theseFormulas
    designHistOptsDict.update({
        thisForm: addExogHistoryTerms[lagSpecIdx]
        for thisForm in theseFormulas})
#
lOfDesignFormulas.append('NULL')
formulasShortHand['NULL'] = 'NULL'
designIsLinear['NULL'] = True
#
templateHistOptsDict = {}
lOfHistTemplates = []
lOfEndogAndExogTemplates = []

for lagSpecIdx in range(len(addEndogHistoryTerms)):
    lagSpec = 'enhto{}'.format(lagSpecIdx)
    wrapperFun = genRcbWrap(lagSpec)
    histTemplate = wrapperFun('{}')
    lOfHistTemplates.append(histTemplate)
    templateHistOptsDict[histTemplate] = addEndogHistoryTerms[lagSpecIdx]
lOfHistTemplates.append('NULL')
#
# exog, self, ensemble
lOfEndogAndExogTemplates = [
    (lOfDesignFormulas[0], lOfHistTemplates[1], lOfHistTemplates[1],), # 0: full exog
    (lOfDesignFormulas[1], lOfHistTemplates[1], lOfHistTemplates[1],), # 1: v only
    (lOfDesignFormulas[2], lOfHistTemplates[1], lOfHistTemplates[1],), # 2: a r only
    (lOfDesignFormulas[0], lOfHistTemplates[1], lOfHistTemplates[0],), # 3: full exog and self
    (lOfDesignFormulas[0], lOfHistTemplates[1], lOfHistTemplates[1],), # 4: full exog and self and ensemble
    (lOfDesignFormulas[1], lOfHistTemplates[1], lOfHistTemplates[1],), # 5: v only and self and ensemble
    (lOfDesignFormulas[2], lOfHistTemplates[1], lOfHistTemplates[1],), # 6: a r only exog and self and ensemble
]
lhsMasksOfInterest = {
    'plotPredictions': [0, 3],
    'varVsEnsemble': [0, 3]
    }
######
################ define model comparisons
# "test" should be the "bigger" model (we are adding coefficients and asking whether they improved performance
modelsToTest = []
    modelsToTest.append({
        'testDesign': 0,
        'refDesign': 1,
        'testCaption': 'v + a + r',
        'refCaption': 'v',
        'captionStr': 'partial R2 of adding terms for A+R to V',
        'testType': 'ARTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[0, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[0, 'lagSpec'],
        })
    modelsToTest.append({
        'testDesign': 0,
        'refDesign': 2,
        'testCaption': 'v + a + r',
        'refCaption': 'a + r',
        'captionStr': 'partial R2 of adding terms for V to A+R',
        'testType': 'VTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[0, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[0, 'lagSpec'],
        })
    modelsToTest.append({
        'testDesign': 3,
        'refDesign': 0,
        'testCaption': 'v + a + r + self',
        'refCaption': 'v + a + r',
        'captionStr': 'partial R2 of adding terms for self to V+A+R',
        'testType': 'selfTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[3, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[3, 'lagSpec'],
        })
    modelsToTest.append({
        'testDesign': 3,
        'refDesign': 0,
        'testCaption': 'v + a + r + self',
        'refCaption': 'v + a + r',
        'captionStr': 'partial R2 of adding terms for self to V+A+R',
        'testType': 'selfTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[3, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[3, 'lagSpec'],
        })
    modelsToTest.append({
        'testDesign': 5,
        'refDesign': 4,
        'testCaption': 'v + a + r + ensemble',
        'refCaption': 'v + ensemble',
        'captionStr': 'partial R2 of adding terms for A+R to V+A+R+Ens',
        'testType': 'AREnsTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[4, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[4, 'lagSpec'],
        })
    modelsToTest.append({
        'testDesign': 6,
        'refDesign': 4,
        'testCaption': 'v + a + r + ensemble',
        'refCaption': 'a + r + ensemble',
        'captionStr': 'partial R2 of adding terms for V to V+A+R+Ens',
        'testType': 'VEnsTerms',
        'testHasEnsembleHistory': lhsMasksInfo.loc[4, 'selfTemplate'] != 'NULL',
        'lagSpec': lhsMasksInfo.loc[4, 'lagSpec'],
        })
burnInPeriod = 600e-3
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