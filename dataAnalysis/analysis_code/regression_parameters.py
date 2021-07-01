
import dataAnalysis.custom_transformers.tdr as tdr


'''lOfDesignFormulas = [
    'velocity + electrode:(rcb(amplitude, **hto0) + electrode:rcb(amplitude * RateInHz, **hto0))',
    'velocity + velocity:electrode:(rcb(amplitude, **hto0) + velocity:electrode:rcb(amplitude * RateInHz, **hto0))',
    'velocity + electrode:(amplitude/RateInHz)',
    'velocity * electrode:(amplitude/RateInHz)', ]'''

regressionColumnsToUse = ['velocity', 'amplitude', 'RateInHz', 'electrode']
regressionColumnRenamer = {
    'velocity': 'v', 'amplitude': 'a', 'RateInHz': 'r', 'electrode': 'e'
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


zeroLagModels = {
    'v': vStr(), 'a': aStr(), 'va': vaStr(), 'ar': arStr(), 'var': varStr()
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
    'var': varStr(genRcbWrap('hto0'))
    }
lOfDesignFormulas = [
    '{v} + {ar} + {var}'.format(**laggedModels),
    '{v} + {a} + {va}'.format(**laggedModels),
    '{v} + {ar}'.format(**laggedModels),
    '{v} + {a}'.format(**laggedModels),
    '{v}'.format(**laggedModels),
    '{ar}'.format(**laggedModels),
    '{a}'.format(**laggedModels),
]

addHistoryTerms = [
    {
        'nb': 5,
        'dt': None,
        'historyLen': 400e-3,
        'b': 0.001, 'useOrtho': True,
        'normalize': True, 'groupBy': 'trialUID',
        'zflag': False, 'logBasis': False,
        'addInputToOutput': True},
    ]

def getHistoryOpts(hTDict, iteratorOpts, rasterOpts):
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
    hTDict['dt'] = binInterval
    return hTDict
