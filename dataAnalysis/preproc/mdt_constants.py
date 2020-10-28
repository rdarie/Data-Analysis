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
    0: 65.104,  # (0.0153 s)
    1: 32.552,  # (0.0307 s)
    2: 16.276,  # (0.614 s)
    3: 8.138,  # (0.123 s)
    4: 4.069,  # (0.245 s)
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