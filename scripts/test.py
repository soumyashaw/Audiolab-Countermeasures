#!/usr/bin/python

"""
Speech Transmission Index (STI) test script

Copyright (C) 2011 Jon Polom <jmpolom@wayne.edu>
Licensed under the GNU General Public License
"""

from datetime import date
from sti import stiFromAudio, readwav

__author__ = "Jonathan Polom <jmpolom@wayne.edu>"
__date__ = date(2011, 4, 22)
__version__ = "0.5"

def testSTI():
    # read audio
    refAudio, refRate = readwav('/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TrialData/OriginalData/common_voice_en_38024625.wav')
    degrAudio, degrRate = readwav('/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TrialData/augmented_data/gaussian_noise/g25dB_common_voice_en_38024625.wav')

    # calculate the STI. Visually verify console output.
    stis = stiFromAudio(refAudio, degrAudio, refRate)
    
    print("Test Result:")
    
    # test result
    if abs(stis - 0.63) < 0.002:
        print("OK")
        return 0
    else:
        print("FAILED")
        return 1

if __name__ == '__main__':
    testSTI()
