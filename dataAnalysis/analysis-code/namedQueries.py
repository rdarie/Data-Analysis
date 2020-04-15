namedQueries = {
    'align': {
        'all': "(t>0)",
        'midPeak': "(pedalMovementCat=='midPeak')",
        'outbound': "(pedalMovementCat=='outbound')",
        'stimOn': "(stimCat=='stimOn')",
        'stimOff': "(stimCat=='stimOff')",
        'outboundXS': "(pedalSizeCat=='XS')&(pedalMovementCat=='outbound')",
        'outboundS': "(pedalSizeCat=='S')&(pedalMovementCat=='outbound')",
        'outboundM': "(pedalSizeCat=='M')&(pedalMovementCat=='outbound')",
        'outboundL': "(pedalSizeCat=='L')&(pedalMovementCat=='outbound')",
        'outboundXL': "(pedalSizeCat=='XL')&(pedalMovementCat=='outbound')",
        'CCW': "(pedalDirection=='CCW')",
        'CW': "(pedalDirection=='CW')",
        'noStim': "(amplitude==0)",
        'RateInHz==50or0Fuzzy': '((RateInHzFuzzy==50)|(RateInHzFuzzy==0))',
        'RateInHz==100or0Fuzzy': '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))',
        'RateInHz==50or0': '((RateInHz==50)|(RateInHz==0))',
        'RateInHz==100or0': '((RateInHz==100)|(RateInHz==0))'
    },
    'unit': {
        'fr': "(chanName.str.endswith('fr#0'))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt#0'))",
        'raster': "(chanName.str.endswith('raster#0'))",
        'all': "(chanName.str.endswith('#0'))",
        'pca': "(chanName.str.contains('pca'))",
        'oech': "(chanName.str.contains('CH'))",
        'isispinaloremg': "((chanName.str.contains('caudal'))or(chanName.str.contains('rostral'))or(chanName.str.contains('Emg')))",
        'isispinal': "( (chanName.str.contains('caudal'))or(chanName.str.contains('rostral')) )",
        'isiemg': "(chanName.str.contains('Emg'))",
        'isiemgenv': "(chanName.str.contains('EmgEnv'))",
        'isiemgraw': "((chanName.str.contains('Emg')) and not (chanName.str.contains('EmgEnv')))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
        'neural': "((chanName.str.contains('elec'))or(chanName.str.contains('nform')))or(chanName.str.contains('utah'))",
        'rig': "not((chanName.str.contains('elec'))or(chanName.str.contains('pca'))or(chanName.str.contains('ainp')))"
    },
    'chan': {
        'all': "(chanName.notna())",
        'fr': "(chanName.str.endswith('fr'))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt'))",
        'raster': "(chanName.str.endswith('raster'))",
        'oech': "(chanName.str.contains('CH'))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
        'notoeaux': "not((chanName.str.contains('AUX')))",
        'rig': "not((chanName.str.contains('elec'))or(chanName.str.contains('pca')))"
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
        namedQueries['align']['RateInHz==100or0'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim100HzCCW': '&'.join([
        namedQueries['align']['RateInHz==100or0'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'outboundWithStim50HzCCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==50or0'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakCCW': '&'.join([
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakCW': '&'.join([
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakM_CCW': '&'.join([
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW'],
        "(pedalSizeCat=='M')"
        ])
    })
namedQueries['align'].update({
    'midPeakM_CW': '&'.join([
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW'],
        "(pedalSizeCat=='M')"
        ])
    })
namedQueries['align'].update({
    'midPeakS': '&'.join([
        namedQueries['align']['midPeak'],
        "(pedalSizeCat=='S')"
        ])
    })
namedQueries['align'].update({
    'midPeakXS': '&'.join([
        namedQueries['align']['midPeak'],
        "(pedalSizeCat=='XS')"
        ])
    })
namedQueries['align'].update({
    'midPeakM': '&'.join([
        namedQueries['align']['midPeak'],
        "(pedalSizeCat=='M')"
        ])
    })
namedQueries['align'].update({
    'midPeakL': '&'.join([
        namedQueries['align']['midPeak'],
        "(pedalSizeCat=='L')"
        ])
    })
namedQueries['align'].update({
    'midPeakXL': '&'.join([
        namedQueries['align']['midPeak'],
        "(pedalSizeCat=='XL')"
        ])
    })
###
namedQueries['align'].update({
    'outboundXS_CW': '&'.join([
        namedQueries['align']['outboundXS'],
        namedQueries['align']['CW'],
        ])
    })
namedQueries['align'].update({
    'outboundXS_CCW': '&'.join([
        namedQueries['align']['outboundXS'],
        namedQueries['align']['CCW'],
        ])
    })
namedQueries['align'].update({
    'outboundS_CW': '&'.join([
        namedQueries['align']['outboundS'],
        namedQueries['align']['CW'],
        ])
    })
namedQueries['align'].update({
    'outboundS_CCW': '&'.join([
        namedQueries['align']['outboundS'],
        namedQueries['align']['CCW'],
        ])
    })
namedQueries['align'].update({
    'outboundM_CW': '&'.join([
        namedQueries['align']['outboundM'],
        namedQueries['align']['CW'],
        ])
    })
namedQueries['align'].update({
    'outboundM_CCW': '&'.join([
        namedQueries['align']['outboundM'],
        namedQueries['align']['CCW'],
        ])
    })
namedQueries['align'].update({
    'outboundL_CW': '&'.join([
        namedQueries['align']['outboundL'],
        namedQueries['align']['CW'],
        ])
    })
namedQueries['align'].update({
    'outboundL_CCW': '&'.join([
        namedQueries['align']['outboundL'],
        namedQueries['align']['CCW'],
        ])
    })
namedQueries['align'].update({
    'outboundXL_CW': '&'.join([
        namedQueries['align']['outboundXL'],
        namedQueries['align']['CW'],
        ])
    })
namedQueries['align'].update({
    'outboundXL_CCW': '&'.join([
        namedQueries['align']['outboundXL'],
        namedQueries['align']['CCW'],
        ])
    })
###
namedQueries['align'].update({
    'midPeakNoStimCCW': '&'.join([
        namedQueries['align']['noStim'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CCW']
        ])
    })
namedQueries['align'].update({
    'midPeakNoStim': '&'.join([
        namedQueries['align']['noStim'],
        namedQueries['align']['midPeak'],
        ])
    })
namedQueries['align'].update({
    'outboundWithStim100HzCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==100or0'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim100HzCW': '&'.join([
        namedQueries['align']['RateInHz==100or0'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'outboundWithStim50HzCW': '&'.join([
        namedQueries['align']['pedalSizeCat>M'],
        namedQueries['align']['RateInHz==50or0'],
        namedQueries['align']['outbound'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakWithStim50HzCW': '&'.join([
        namedQueries['align']['RateInHz==50or0'],
        namedQueries['align']['midPeak'],
        namedQueries['align']['CW']
        ])
    })
namedQueries['align'].update({
    'midPeakNoStimCW': '&'.join([
        namedQueries['align']['noStim'],
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
