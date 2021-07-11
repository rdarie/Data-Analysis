
import dataAnalysis.custom_transformers.tdr as tdr


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

laggedModels = {
    'v': vStr(genRcbWrap('hto0')), 'a': aStr(genRcbWrap('hto0')),
    'r': rStr(genRcbWrap('hto0')),
    'va': vaStr(genRcbWrap('hto0')), 'ar': arStr(genRcbWrap('hto0')),
    'vr': vrStr(genRcbWrap('hto0')),
    'var': varStr(genRcbWrap('hto0'))
    }

'''lOfDesignFormulas = [
    '{v} + {ar} + {var}'.format(**laggedModels),
    '{v} + {a} + {va}'.format(**laggedModels),
    '{v} + {ar}'.format(**laggedModels),
    '{v} + {a}'.format(**laggedModels),
    '{v}'.format(**laggedModels),
    '{ar}'.format(**laggedModels),
    '{a}'.format(**laggedModels),
    '{v} + {a} + {r} + {va} + {ar} + {vr} + {var}'.format(**laggedModels),
    '{v} + {a} + {r}'.format(**laggedModels),
    ]'''
lOfDesignFormulas = [
    '{v} + {a} + {r} + {va} + {ar} + {vr} + {var} - 1'.format(**laggedModels),
    '{v} + {a} + {r} - 1'.format(**laggedModels),
    '{a} + {r} - 1'.format(**laggedModels),
    '{v} - 1'.format(**laggedModels),
    ]

lOfEnsembleTemplates = ['rcb({}, **hto0)']

lOfSelfTemplates = ['rcb({}, **hto0)']

addHistoryTerms = [
    {
        'nb': 5,
        'dt': None,
        'historyLen': 200e-3,
        'b': 0.001, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False, 'logBasis': False,
        'addInputToOutput': False},
    ]

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
            'captionStr': 'partial R2 of allowing interactions between regression terms, vs assuming their independence'
        },]