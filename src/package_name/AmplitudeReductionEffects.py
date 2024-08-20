#!/usr/bin/python

import numpy as np

def find_volume(audio):
    rms = np.sqrt(np.mean(audio**2))

    # Convert the RMS value to decibels (dB)
    return 20 * np.log10(rms)