namedQueries = {
    'align': {
        'midPeak': "(pedalMovementCat=='midPeak')",
        'outbound': "(pedalMovementCat=='outbound')",
        'stimOn': "(stimCat=='stimOn')",
        'CCW': "(pedalDirection=='CCW')",
        'CW': "(pedalDirection=='CW')",
        'RateInHz==50or0Fuzzy': '((RateInHzFuzzy==50)|(RateInHzFuzzy==0))',
        'RateInHz==100or0Fuzzy': '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))'
    },
    'unit': {
        'fr': "(chanName.str.endswith('fr#0'))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt#0'))",
        'all': "(chanName.str.endswith('#0'))",
        'pca': "(chanName.str.contains('pca'))",
        'oech': "(chanName.str.contains('CH'))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
        'rig': "not((chanName.str.contains('elec'))or(chanName.str.contains('pca')))"
    },
    'chan': {
        'fr': "(chanName.str.endswith('fr'))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt'))",
        'raster': "(chanName.str.endswith('raster'))",
        'oech': "(chanName.str.contains('CH'))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
    }
}
namedQueries['align'].update({
    'pedalSizeCat>S': '(' + '|'.join([
        '(pedalSizeCat == \'{}\')'.format(i)
        for i in ['S', 'M', 'L', 'XL']
        ]) + ')'
    })
namedQueries['align'].update({
    'pedalSizeCat>M': '(' + '|'.join([
        '(pedalSizeCat == \'{}\')'.format(i)
        for i in ['M', 'L', 'XL']
        ]) + ')'
    })
namedQueries['align'].update({
    'outboundWithStim100HzCCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==100or0Fuzzy'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim100HzCCW': '&'.join([
        namedQueries['align']['RateInHz==100or0Fuzzy'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'outboundWithStim50HzCCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==50or0Fuzzy'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim50HzCCW': '&'.join([
        namedQueries['align']['RateInHz==50or0Fuzzy'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'outboundWithStim100HzCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==100or0Fuzzy'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim100HzCW': '&'.join([
        namedQueries['align']['RateInHz==100or0Fuzzy'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'outboundWithStim50HzCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==50or0Fuzzy'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim50HzCW': '&'.join([
        namedQueries['align']['RateInHz==50or0Fuzzy'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'stimOnLowRate': '&'.join([
        namedQueries['align']['stimOn'],
        "(RateInHz < 5)"
        ])
    })
