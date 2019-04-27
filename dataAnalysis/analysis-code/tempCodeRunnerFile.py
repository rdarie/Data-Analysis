
    #  calculate movement durations
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedBase').sum())