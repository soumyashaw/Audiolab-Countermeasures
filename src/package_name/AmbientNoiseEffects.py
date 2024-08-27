#!/usr/bin/python

import os
import random
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
from package_name.sti import stiFromAudio
from package_name.utils import make_directory

def add_ambient_noise(audioPath, noisePath, snr_dB, sti_threshold):
    # Load the original audio file
    signal, sr = librosa.load(audioPath, sr=None)

    # Load the noise file
    noise_signal, noise_sr = librosa.load(noisePath, sr=None)

    # Resample the noise signal to match the sampling rate of the original signal
    noise_signal = librosa.resample(noise_signal, orig_sr=noise_sr, target_sr=sr)

    # Ensure the noise signal is at least as long as the original signal
    if len(noise_signal) < len(signal):
        # Repeat the noise signal to match the length of the original signal
        repetitions = int(np.ceil(len(signal) / len(noise_signal)))
        noise_signal = np.tile(noise_signal, repetitions)[:len(signal)]

    else:
        # Trim the noise signal to match the length of the original signal
        noise_signal = noise_signal[:len(signal)]

    # Calculate the power of the signal
    signal_power = np.sum(signal ** 2) / len(signal)

    # Calculate the power of the noise signal
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)

    flag_fault = True

    while flag_fault:

        # Calculate the desired noise power based on the desired SNR
        snr_linear = 10 ** (snr_dB / 10.0)
        desired_noise_power = signal_power / snr_linear

        # Scale the noise signal to achieve the desired noise power
        scaled_noise = noise_signal * np.sqrt(desired_noise_power / noise_power)

        # Add the noise to the signal
        noisy_signal = signal + scaled_noise

        if sr <=16000:
            num_samples = int(len(signal) * float(18000) / sr)
            signal = resample(signal, num_samples)
            noisy_signal = resample(noisy_signal, num_samples)
            sr = 18000

        # Calculate the STI of the noisy signal
        STI = stiFromAudio(signal, noisy_signal, sr)

        if STI > sti_threshold:
            flag_fault = False
        else:
            snr_dB += 5
            flag_fault = True

    return noisy_signal, sr

def add_ambient_noise_effects(SNR_levels_dB, reference_dir, ambient_noise_dir, sti_threshold):
    output_files = []

    print(" "*50 + "\033[91mAdding Ambient Noise\033[0m")
    print()

    # Enumerate the ambient noise files in the directory
    noise_files = os.listdir(ambient_noise_dir)

    # Declare the SNR levels for the ambient noise
    

    # List all the audio files in the reference directory (original audio files)
    audio_files = os.listdir(reference_dir)

    # Change the directory to the reference directory
    os.chdir(reference_dir)

    os.chdir("../")

    # Make a directory to store the augmented data
    make_directory(os.getcwd() + "/augmented_data/ambient_noise/")

    target_dir = os.getcwd() + "/augmented_data/ambient_noise/"

    for audio in tqdm(audio_files, desc="Adding Ambient Noise to Audios"):
        # Randomly choose the ambient noise file
        noise = random.choice(noise_files)

        # Randomly choose the SNR level
        SNR = random.choice(SNR_levels_dB)

        input_audio = reference_dir + str(audio)

        noise_audio = ambient_noise_dir + str(noise)

        # Append the identifier string to output audio file
        output_audio = f"{target_dir}amb{str(SNR)}dB_{str(audio)}"

        # Append the output audio file to the list for text file creation
        output_files.append("amb" + str(SNR) + "dB_" + str(audio))

        # Call the function to add white noise to the audio file
        noisy_signal, sample_rate = add_ambient_noise(input_audio, noise_audio, SNR, sti_threshold)

        # Save the output with noise to a new file
        sf.write(output_audio, noisy_signal, sample_rate)

    print()
    print("\033[92mAmbient Noise added successfully!\033[0m")

    if len(output_files) > 0:
        # Create a text file to store the output audio files
        os.chdir(reference_dir)
        os.chdir("../")

        with open('augmented_data/ambient_noise.txt', 'w') as file:
            for item in output_files:
                file.write(f"{item}\n")
