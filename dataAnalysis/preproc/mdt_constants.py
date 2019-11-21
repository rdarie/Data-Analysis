from enum import Enum

sampleRate = {
    0: 250,
    1: 500,
    2: 1000,
    240: None
    }

cycleUnits = {
    # ushort, convert to seconds
    0: 100e-3,
    1: 1,
    2: 10,
    }

muxIdx = {
    0: None,
    1: 0,
    2: 1,
    4: 2,
    8: 3,
    16: 4,
    32: 5,
    64: 6,
    128: 7
    }
