from enum import Enum

TdSampleRates = {
    # Hz
    0: 250,
    1: 500,
    2: 1000,
    240: None
    }

AccelSampleRate = {
    # Hz
    0: 64,
    1: 32,
    2: 16,
    3: 8,
    4: 4,
    255: None
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

streamingFrameRate = {
    3: 30,
    4: 40,
    5: 50,
    6: 60,
    7: 70,
    8: 80,
    9: 90,
    10: 100
}