namedQueries = {
    'align': {
        'all': "(t>0)",
        'midPeak': "(pedalMovementCat=='midPeak')",
        'outbound': "(pedalMovementCat=='outbound')",
        'return': "(pedalMovementCat=='return')",
        'reachedPeak': "(pedalMovementCat=='reachedPeak')",
        'reachedBase': "(pedalMovementCat=='reachedBase')",
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
        'RateInHz==100or0': '((RateInHz==100)|(RateInHz==0))',
        'RateInHz>20or0': '((RateInHz>20)|(RateInHz==0))',
    },
    'unit': {
        'fr': "(chanName.str.endswith('fr#0'))",
        'utahlfp': "(chanName.str.contains('elec')and(not(chanName.str.endswith('fr#0'))))",
        'lfp': "((chanName.str.contains('elec') or chanName.str.contains('utah') or chanName.str.contains('nform')) and not(chanName.str.endswith('fr#0') or chanName.str.contains('rawAverage') or chanName.str.contains('deviation') or chanName.str.contains('_artifact') or chanName.str.contains('outlierMask')))",
        'derivedFromLfp': "((chanName.str.contains('elec') or chanName.str.contains('utah') or chanName.str.contains('nform')) and (chanName.str.endswith('fr#0') or chanName.str.contains('rawAverage') or chanName.str.contains('deviation') or chanName.str.contains('_artifact') or chanName.str.contains('outlierMask')))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt#0'))",
        'raster': "(chanName.str.endswith('raster#0'))",
        'all': "(chanName.str.endswith('#0'))",
        'pca': "(chanName.str.contains('pca'))",
        'oech': "(chanName.str.contains('CH'))",
        'isispinaloremg': "((chanName.str.contains('caudal'))or(chanName.str.contains('rostral'))or(chanName.str.contains('Emg')))",
        'isispinal': "( (chanName.str.contains('caudal'))or(chanName.str.contains('rostral')) )",
        'isiemg': "(chanName.str.contains('Emg'))",
        'isiacc': "(chanName.str.contains('Acc'))",
        'isiemgoracc': "(chanName.str.contains('Acc'))or(chanName.str.contains('Emg'))",
        'isiemgenv': "(chanName.str.contains('EmgEnv'))",
        'isiemgraw': "((chanName.str.contains('Emg')) and not (chanName.str.contains('EmgEnv')))",
        'isiemgenvoraccorspinal': "(chanName.str.contains('Acc'))or(chanName.str.contains('EmgEnv'))or(chanName.str.contains('caudal'))or(chanName.str.contains('rostral'))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
        'neural': "((chanName.str.contains('elec'))or(chanName.str.contains('nform')))or(chanName.str.contains('utah'))",
        'rig': "not((chanName.str.contains('elec'))or(chanName.str.contains('pca'))or(chanName.str.contains('nform'))or(chanName.str.contains('ainp')))",
        'jointAngle': "chanName.isin(['right_hip_angle#0', 'right_knee_angle#0', 'right_ankle_angle#0'])",
        'jointAngularVelocity': "chanName.isin(['right_hip_omega#0', 'right_knee_omega#0', 'right_ankle_omega#0'])",
        'jointAngularVelocityMagnitude': "chanName.isin(['right_hip_omega_abs#0', 'right_knee_omega_abs#0', 'right_ankle_omega_abs#0'])",
        'endpointForce': "chanName.isin(['forceX#0', 'forceY#0'])",
        'endpointForceMagnitude': "chanName.isin(['forceMagnitude#0', 'forceX_abs#0', 'forceY_abs#0'])",
        'endpointYank': "chanName.isin(['forceX_prime#0', 'forceY_prime#0', 'forceMagnitude_prime#0'])",
        'endpointYankMagnitude': "chanName.isin(['forceX_prime_abs#0', 'forceY_prime_abs#0'])",
        'pedalPosition': "chanName.isin(['position#0'])",
        'pedalPositionXY': "chanName.isin(['position_x#0', 'position_y#0'])",
        'pedalVelocity': "chanName.isin(['velocity#0'])",
        'pedalVelocityXY': "chanName.isin(['velocity_x#0', 'velocity_y#0'])",
    },
    'chan': {
        'all': "(chanName.notna())",
        'lfp': "((chanName.str.contains('elec') or chanName.str.contains('utah') or chanName.str.contains('nform')) and not(chanName.str.contains('rawAverage') or chanName.str.contains('deviation') or chanName.str.contains('_artifact') or chanName.str.contains('outlierMask')))",
        'derivedFromLfp': "((chanName.str.contains('elec') or chanName.str.contains('utah') or chanName.str.contains('nform')) and (chanName.str.contains('rawAverage') or chanName.str.contains('deviation') or chanName.str.contains('_artifact') or chanName.str.contains('outlierMask')))",
        'fr': "((chanName.str.contains('elec')or(chanName.str.contains('utah'))or(chanName.str.contains('nform')))and((chanName.str.endswith('fr'))))",
        'fr_sqrt': "(chanName.str.endswith('fr_sqrt'))",
        'raster': "((chanName.str.contains('elec')or(chanName.str.contains('utah'))or(chanName.str.contains('nform')))and((chanName.str.endswith('raster'))))",
        'oech': "(chanName.str.contains('CH'))",
        'oechorsense': "((chanName.str.contains('CH'))or(chanName.str.contains('Sense')))",
        'oechorins': "((chanName.str.contains('CH'))or(chanName.str.contains('ins')))",
        'notoeaux': "not((chanName.str.contains('AUX')))",
        'rig': "not((chanName.str.contains('elec'))or(chanName.str.contains('utah'))or(chanName.str.contains('pca'))or(chanName.str.contains('nform'))or(chanName.str.contains('utah')))",
        'notainp': "not((chanName.str.contains('ainp'))or(chanName.str.contains('analog')))",
        'isispinaloremg': "((chanName.str.contains('caudal'))or(chanName.str.contains('rostral'))or(chanName.str.contains('Emg')))",
        'isispinal': "( (chanName.str.contains('caudal'))or(chanName.str.contains('rostral')) )",
        'isiemgoranalog': "((chanName.str.contains('Emg'))|(chanName.str.contains('Analog')))",
        'isiemg': "(chanName.str.contains('Emg'))",
        'isiacc': "(chanName.str.contains('Acc'))",
        'isiemgoracc': "(chanName.str.contains('Acc'))or(chanName.str.contains('Emg'))",
        'isiemgenvoraccorspinal': "(chanName.str.contains('Acc'))or(chanName.str.contains('EmgEnv'))or(chanName.str.contains('caudal'))or(chanName.str.contains('rostral'))",
        'isiemgenv': "(chanName.str.contains('EmgEnv'))",
        'isiemgraw': "((chanName.str.contains('Emg')) and not (chanName.str.contains('EmgEnv')))",
    }
}
namedQueries['unit'].update({
    'limbState': '(' + '|'.join([
            namedQueries['unit'][key]
            for key in [
                'jointAngle', 'jointAngularVelocity',
                'jointAngularVelocityMagnitude',
                'endpointForce', 'endpointYank',
                'endpointYankMagnitude', 'endpointForceMagnitude'
            ]
        ]) + ')'
    })
namedQueries['unit'].update({
    'pedalState': '(' + '|'.join([
            namedQueries['unit'][key]
            for key in [
                'pedalPosition', 'pedalVelocity',
                'pedalPositionXY', 'pedalVelocityXY',
                'endpointForce', 'endpointYank',
                'endpointYankMagnitude', 'endpointForceMagnitude'
            ]
        ]) + ')'
    })
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
    'starting': '|'.join([
        namedQueries['align']['outbound'],
        namedQueries['align']['return'],
        ])
    })
namedQueries['align'].update({
    'stopping': '|'.join([
        namedQueries['align']['reachedPeak'],
        namedQueries['align']['reachedBase'],
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
listOfSubQ = [
    namedQueries['align']['RateInHz>20or0'],
    namedQueries['align']['outbound'],
    namedQueries['align']['CW']
    ]
namedQueries['align'].update({
    'outboundStim>20HzCW': '&'.join(['({})'.format(subQ) for subQ in listOfSubQ])
    })
listOfSubQ = [
    namedQueries['align']['RateInHz>20or0'],
    namedQueries['align']['starting'],
    namedQueries['align']['CW']
    ]
namedQueries['align'].update({
    'startingStim>20HzCW': '&'.join(['({})'.format(subQ) for subQ in listOfSubQ])
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
namedQueries['align'].update({
    'stimOnLessThan11Hz': '&'.join([
        namedQueries['align']['stimOn'],
        "(RateInHz < 11) & (RateInHz > 5)"
        ])
    })
namedQueries['align'].update({
    'stimOnLessThan30Hz': '&'.join([
        namedQueries['align']['stimOn'],
        "(RateInHz < 30) & (RateInHz > 5)"
        ])
    })
namedQueries['align'].update({
    'stimOnHighRate': '&'.join([
        namedQueries['align']['stimOn'],
        "(RateInHz > 20)"
        ])
    })
namedQueries['align'].update({
    'stimOffHighRate': '&'.join([
        namedQueries['align']['stimOff'],
        "(RateInHz > 20)"
        ])
    })
