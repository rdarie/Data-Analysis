
import dataAnalysis.custom_transformers.tdr as tdr
import pdb
from itertools import product

validTargetLhsMaskIdx = {
    'ols_select_baseline': [0, 3, 4, 7, 8, 9],
    'ols2_select2_baseline': [0, 3, 4, 5, 6],
    'ols_select_spectral_baseline': [0, 3, 4, 7, 8, 9],
    'ols2_select2_spectral_baseline': [0, 3, 4, 5, 6]
}

processSlurmTaskCountPLS = 3
processSlurmTaskCount = 23
processSlurmTaskCountTransferFunctions = 23
joblibBackendArgs = dict(
    backend='loky',
    n_jobs=-1
    )

burnInPeriod = 600e-3

addEndogHistoryTerms = [
    # enhto0
    {
        'nb': 5, 'logBasis': True,
        'dt': None,
        'historyLen': 600e-3,
        'timeDelay': 0.,
        'b': 100e-3, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]
addExogHistoryTerms = [
    # exhto0
    {
        'nb': 5, 'logBasis': True,
        'dt': None,
        'historyLen': 600e-3,
        'timeDelay': 0.,
        'b': 100e-3, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False, 'verbose': 0,
        'joblibBackendArgs': joblibBackendArgs, 'convolveMethod': 'auto'},
    ]

regressionColumnsToUse = [
    # 'velocity_abs',
    'velocity_x', 'velocity_y',
    # 'acceleration_xy',
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
    'acceleration_xy': 'accxy',
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

interactionList = [
    '{' + '{}*{}'.format(aT, bT) + '}'
    for aT, bT in product(['vx', 'vy'], ['a', 'r'])
    ]
templateBothInteractions = (
    '{vx} + {vy} + {a} + {r} + {a*r} + ' +
    ' + '.join(interactionList) + ' - 1')
templateVARInteractions = (
    '{vx} + {vy} + {a} + {r} + ' +
    ' + '.join(interactionList) + ' - 1')

templateIsLinear = {
    '{vx} + {vy} + {a} + {r} - 1': True, # 0
    '{vx} + {vy} - 1': True, # 1
    '{a} + {r} - 1': True, # 2
    templateBothInteractions: False, # 3
    '{vx} + {vy} + {a} + {r} + {a*r} - 1': False, # 4
    templateVARInteractions: False # 5
    }
designFormulaTemplates = [k for k in templateIsLinear.keys()]
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
    for source in ['vx', 'vy']:
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
    theseFormulas = []
    for dft in designFormulaTemplates:
        formattedFormula = dft.format(**laggedModels)
        print(formattedFormula)
        theseFormulas.append(formattedFormula)
        designIsLinear[formattedFormula] = templateIsLinear[dft]
        formulasShortHand[formattedFormula] = '({}, **{})'.format(dft, lagSpec)
    # theseFormulas = [dft.format(**laggedModels) for dft in designFormulaTemplates]
    # formulasShortHand.update({
    #     dft.format(**laggedModels): '({}, **{})'.format(dft, lagSpec)
    #     for dft in designFormulaTemplates})
    # for fIdx in range(len(theseFormulas)):
    #     designIsLinear.update({
    #         theseFormulas[fIdx]: True})
    masterExogFormulas.append(theseFormulas[3])
    for tf in theseFormulas:
        masterExogLookup[tf] = theseFormulas[3]
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
# exog, ensemble, self
lOfEndogAndExogTemplates = [
    (lOfDesignFormulas[0],  lOfHistTemplates[-1], lOfHistTemplates[-1],), # 0: full exog no interactions
    (lOfDesignFormulas[1],  lOfHistTemplates[-1], lOfHistTemplates[-1],), # 1: v only
    (lOfDesignFormulas[2],  lOfHistTemplates[-1], lOfHistTemplates[-1],), # 2: a r only
    #
    (lOfDesignFormulas[0],  lOfHistTemplates[-1], lOfHistTemplates[0],), # 3: full exog and self
    (lOfDesignFormulas[-1], lOfHistTemplates[-1], lOfHistTemplates[0],), # 4: self only
    #
    (lOfDesignFormulas[0], lOfHistTemplates[0],  lOfHistTemplates[0],), # 5: full exog and self and ensemble
    (lOfDesignFormulas[0], lOfHistTemplates[0],  lOfHistTemplates[-1],), # 6: full exog and ensemble
    #
    (lOfDesignFormulas[3], lOfHistTemplates[-1],  lOfHistTemplates[0],), # 7: full exog, self, all interactions
    (lOfDesignFormulas[4], lOfHistTemplates[-1],  lOfHistTemplates[0],), # 8: full exog, self,  a*r interactions
    (lOfDesignFormulas[5], lOfHistTemplates[-1],  lOfHistTemplates[0],), # 9: full exog, self, (v, stim) interactions
    #
    (lOfDesignFormulas[3], lOfHistTemplates[0],  lOfHistTemplates[0],), # 10: full exog, self and ensemble, interactions and 
    ]
lhsMasksOfInterest = {
    'plotPredictions': [0, 4, 7],
    'varVsEnsemble': [0, 4, 7]
    }
# ######
################ define model comparisons
# "test" should be the "bigger" model (we are adding coefficients and asking whether they improved performance
modelsToTestStr = '''
modelsToTest = []
modelsToTest.append({
    'testDesign': 0,
    'refDesign': 1,
    'testCaption': 'v + a + r',
    'refCaption': 'v',
    'captionStr': 'partial R2 of adding terms for A+R to V',
    'testType': 'ARTerms',
    'testHasEnsembleHistory': False,
    'lagSpec': lhsMasksInfo.loc[0, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 0,
    'refDesign': 2,
    'testCaption': 'v + a + r',
    'refCaption': 'a + r',
    'captionStr': 'partial R2 of adding terms for V to A+R',
    'testType': 'VTerms',
    'testHasEnsembleHistory': False,
    'lagSpec': lhsMasksInfo.loc[0, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 3,
    'refDesign': 4,
    'testCaption': 'v + a + r + self',
    'refCaption': 'self',
    'captionStr': 'partial R2 of adding terms for V+A+R to V+A+R+self',
    'testType': 'exogVSExogAndSelf',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[3, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 3,
    'refDesign': 0,
    'testCaption': 'v + a + r + self',
    'refCaption': 'v + a + r',
    'captionStr': 'partial R2 of adding terms for self to V+A+R+self',
    'testType': 'selfVSExogAndSelf',
    'testHasEnsembleHistory': False,
    'lagSpec': lhsMasksInfo.loc[3, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 5,
    'refDesign': 6,
    'testCaption': 'v + a + r + self + ensemble',
    'refCaption': 'v + a + r + ensemble',
    'captionStr': 'partial R2 of adding terms for self to V+A+R+self+ens',
    'testType': 'selfVSFull',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[5, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 5,
    'refDesign': 3,
    'testCaption': 'v + a + r + self + ensemble',
    'refCaption': 'v + a + r + self',
    'captionStr': 'partial R2 of adding terms for ens to V+A+R+self+ens',
    'testType': 'ensVSFull',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[5, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 7,
    'refDesign': 3,
    'testCaption': 'v + a + r + self + (all interactions)',
    'refCaption': 'v + a + r + self',
    'captionStr': 'partial R2 of adding terms for all interactions to V+A+R+self',
    'testType': 'VARVsVARInter',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[7, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 7,
    'refDesign': 8,
    'testCaption': 'v + a + r + self + (all interactions)',
    'refCaption': 'v + a + r + self + (a*r)',
    'captionStr': 'partial R2 of adding terms for (v,stim) to V+A+R+(a*r)',
    'testType': 'VStimInterVsVARInter',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[7, 'lagSpec'],
    })
modelsToTest.append({
    'testDesign': 7,
    'refDesign': 9,
    'testCaption': 'v + a + r + self + (all interactions)',
    'refCaption': 'v + a + r + self + (v,stim)',
    'captionStr': 'partial R2 of adding terms for (a*r) to V+A+R+(v,stim)',
    'testType': 'ARInterVsVARInter',
    'testHasEnsembleHistory': True,
    'lagSpec': lhsMasksInfo.loc[7, 'lagSpec'],
    })
    '''
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