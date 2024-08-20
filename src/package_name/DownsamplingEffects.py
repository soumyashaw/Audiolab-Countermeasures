#!/usr/bin/python

import librosa
from scipy.signal import resample

def downsample_audio(audioPath, sampling_freq, original_sampling_freq = 44100):
    # Load the audio file
    audio, sr = librosa.load(audioPath, sr=None)

    # Downsample the audio file
    number_of_samples = round(len(audio) * float(sampling_freq) / original_sampling_freq)
    downsampled_audio = resample(audio, number_of_samples)

    # Upsample the audio file to the original sampling frequency
    number_of_samples = round(len(downsampled_audio) * float(original_sampling_freq) / sampling_freq)
    audio = resample(downsampled_audio, number_of_samples)

    return audio, original_sampling_freq