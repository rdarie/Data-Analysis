
import dataAnalysis.custom_transformers.tdr as tdr

addHistoryTerms = [
    {
        'nb': 5,
        'dt': None,
        'historyLen': 250e-3,
        'b': 0.001, 'useOrtho': False,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False, 'logBasis': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False},
    {
        'nb': 5,
        'dt': None,
        'historyLen': 250e-3,
        'b': 0.001, 'useOrtho': False,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False, 'logBasis': True,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False},
    {
        'nb': 10,
        'dt': None,
        'historyLen': 500e-3,
        'b': 0.001, 'useOrtho': False,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False, 'logBasis': False,
        'causalShift': True, 'causalFill': True,
        'addInputToOutput': False},
    ]
'''lOfDesignFormulas = [
    'velocity + electrode:(rcb(amplitude, **hto0) + electrode:rcb(amplitude * RateInHz, **hto0))',
    'velocity + velocity:electrode:(rcb(amplitude, **hto0) + velocity:electrode:rcb(amplitude * RateInHz, **hto0))',
    'velocity + electrode:(amplitude/RateInHz)',
    'velocity * electrode:(amplitude/RateInHz)', ]'''

regressionColumnsToUse = ['velocity_abs', 'amplitude', 'RateInHz', 'electrode']
regressionColumnRenamer = {
    'velocity_abs': 'v', 'amplitude': 'a', 'RateInHz': 'r', 'electrode': 'e'
    }

def iWrap(x):
    return 'I({})'.format(x)


def vStr(
        modFun=iWrap):
    return '{}'.format(modFun('v'))


def aStr(
        modFun=iWrap):
    return 'e:{}'.format(modFun('a'))


def rStr(
        modFun=iWrap):
    return 'e:{}'.format(modFun('r'))


def vaStr(
        modFun=iWrap):
    return 'e:{}'.format(modFun('v*a'))


def varStr(
        modFun=iWrap):
    return 'e:({}+{})'.format(modFun('v*a'), modFun('v*a*r'))


def arStr(
        modFun=iWrap):
    return 'e:({}+{})'.format(modFun('a'), modFun('a*r'))

def vrStr(
        modFun=iWrap):
    return 'e:({}+{})'.format(modFun('r'), modFun('v*r'))


zeroLagModels = {
    'v': vStr(), 'a': aStr(), 'r': rStr(), 'va': vaStr(), 'ar': arStr(), 'vr': vrStr(), 'var': varStr()
}

rcb = tdr.patsyRaisedCosTransformer

def genRcbWrap(htoStr):
    def rcbWrap(x):
        return 'rcb({},**{})'.format(x, htoStr)
    return rcbWrap

def ddt(x):
    return x.diff().fillna(0)

def diffWrap(x):
    return('ddt({})'.format(x))

def abv(x):
    return x.abs()

def absWrap(x):
    return('abv({})'.format(x))

lOfDesignFormulas = []
lOfHistTemplates = []
sourceTermDict = {}
sourceHistOptsDict = {}
templateHistOptsDict = {}
designIsLinear = {}
for lagSpecIdx in range(len(addHistoryTerms)):
    lagSpec = 'hto{}'.format(lagSpecIdx)
    wrapperFun = genRcbWrap(lagSpec)
    for source in ['v']:
        sourceTermDict[wrapperFun(source)] = source
        sourceHistOptsDict[wrapperFun(source)] = addHistoryTerms[lagSpecIdx]
    for source in ['a', 'r', 'a*r', 'v*r', 'v*a', 'v*a*r']:
        sourceTermDict['e:'+wrapperFun(source)] = source
        sourceHistOptsDict['e:'+wrapperFun(source)] = addHistoryTerms[lagSpecIdx]
    #
    laggedModels = {
        'v': vStr(genRcbWrap(lagSpec)), 'a': aStr(genRcbWrap(lagSpec)),
        'r': rStr(genRcbWrap(lagSpec)),
        'va': vaStr(genRcbWrap(lagSpec)), 'ar': arStr(genRcbWrap(lagSpec)),
        'vr': vrStr(genRcbWrap(lagSpec)),
        'var': varStr(genRcbWrap(lagSpec))
        }
    theseFormulas = [
        # '{v} + {a} + {r} + {va} + {ar} + {vr} + {var} - 1'.format(**laggedModels),
        '{v} + {a} + {r} - 1'.format(**laggedModels),
        '{a} + {r} - 1'.format(**laggedModels),
        '{v} - 1'.format(**laggedModels),
        ]
    designIsLinear.update({
        theseFormulas[0]: True,
        theseFormulas[1]: True,
        theseFormulas[2]: True,
        })
    lOfDesignFormulas += theseFormulas
    histTemplate = 'rcb({}, **{})'.format('{}', lagSpec)
    lOfHistTemplates.append(histTemplate)
    templateHistOptsDict[histTemplate] = addHistoryTerms[lagSpecIdx]
#
lOfEnsembleTemplates = [
    (hT, hT) for hT in lOfHistTemplates
    ]

burnInPeriod = 500e-3
def getHistoryOpts(hTDict, iteratorOpts, rasterOpts):
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
    hTDict['dt'] = binInterval
    return hTDict
# test should be the "bigger" model (we are adding coefficients and asking whether they improved performance
'''
modelsToTest = [
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0))',
            'captionStr': 'partial R2 of allowing pedal velocity to modulate electrode coefficients, vs assuming their independence'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0)',
            'captionStr': 'partial R2 of including any electrode coefficients'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'e:(rcb(a,**hto0)+rcb(a*r,**hto0))',
            'captionStr': 'partial R2 of including any pedal velocity coefficients'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0) + e:rcb(a,**hto0) + e:rcb(v*a,**hto0)',
            'captionStr': 'partial R2 of including a term for modulation of electrode coefficients by RateInHz'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': None,
            'captionStr': 'R2 of the full model, V + AR + VAR'
        },
    ]'''
modelsToTest = [
        {
            'testDesign': lOfDesignFormulas[0],
            'refDesign': lOfDesignFormulas[1],
            'captionStr': 'partial R2 of adding a term for velocity'
        },]