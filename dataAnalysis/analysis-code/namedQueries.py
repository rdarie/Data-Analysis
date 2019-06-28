pedalSizeGreaterThanM = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'

namedQueries = {
    'align': {
        'midPeak': "(pedalMovementCat=='midPeak')",
        'outbound': "(pedalMovementCat=='outbound')",
        'pedalSize>M': pedalSizeGreaterThanM,
        'RateInHz>100or0Fuzzy': '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))'
    },
    'unit': {
        'fr': "(chanName.str.endswith('fr#0'))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt#0'))",
        'pca': "(chanName.str.contains('pca'))"
    }
}
namedQueries['align'].update({'outboundWithStim': '&'.join([
    namedQueries['align']['pedalSize>M'],
    namedQueries['align']['RateInHz>100or0Fuzzy'],
    namedQueries['align']['outbound']
])})