#!/usr/bin/python

import os
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
from package_name.utils import divide_list_randomly, make_directory


def find_volume(audio):
    rms = np.sqrt(np.mean(audio**2))

    # Convert the RMS value to decibels (dB)
    return 20 * np.log10(rms)

def add_amplitude_reduction_effects(reference_dir: str, volume_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mAdding Muffling (Volume Reduction)\033[0m")
    print()

    audio_files = os.listdir(reference_dir)
    audio = random.choice(audio_files)

    # Load the audio file
    reference_audio, sr = librosa.load(reference_dir + audio, sr=None)

    target_rate = 16000

    # Modify sampling rate to 16kHz (for STI calculation)
    if sr != target_rate:
        number_of_samples = round(len(reference_audio) * float(target_rate) / sr)
        reference_audio = resample(reference_audio, number_of_samples)
        sr = target_rate

    vol_dB = 100.0
    dB_reduced = 1

    while vol_dB > volume_threshold:
        db_reduction = -1 * dB_reduced
        reduction_factor = 10 ** (db_reduction / 20)

        volume_reduced_audio = reference_audio * reduction_factor

        vol_dB = find_volume(volume_reduced_audio)

        if vol_dB >= volume_threshold:
            dB_reduced += 1

    vol_dBs = [float(i) for i in range(1, dB_reduced)]
    
    directories_made = []

    # List all the audio files in the reference directory (original audio files)
    reference_files = os.listdir(reference_dir)

    # Divide the list of audio files into n partitions (based on the number of SNR levels)
    audio_files = divide_list_randomly(reference_files, len(vol_dBs))

    # Check the existence of directory to store the augmented data exists
    for i in range(len(vol_dBs)):
        # Change the directory to the reference directory
        os.chdir(reference_dir)

        # Create a new directory to store the augmented data
        os.chdir("../")
        target_dir = os.getcwd() + "/augmented_data/vol_reduction_" + str(int(vol_dBs[i])) + "dB/"
        make_directory(target_dir)
        directories_made.append(target_dir)

        # Reduce the volume of the audio files with given dB levels
        for audio in tqdm(audio_files[i], desc="Reducing Volume in Partition " + str(i+1)):
            input_audio = reference_dir + str(audio)
            # Append the identifier string to output audio file
            output_audio = target_dir + "vol" + str(int(vol_dBs[i])) + "dB_" + str(audio)

            # Append the output audio file to the list for text file creation
            output_files.append("vol" + str(int(vol_dBs[i])) + "dB_" + str(audio))

            reference_audio, sr = librosa.load(input_audio, sr=None)

            db_reduction = -1 * vol_dBs[i]
            reduction_factor = 10 ** (db_reduction / 20)

            volume_reduced_audio = reference_audio * reduction_factor

            sf.write(output_audio, volume_reduced_audio, sr)

        print()

    print("\033[92mVolume Reduced successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(reference_dir)
    os.chdir("../")

    with open('augmented_data/volume_reduction.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")

    # Cleanup: Merge the directories into one
    current_path = os.getcwd() + "/augmented_data/"
    make_directory(current_path + "volume_reduction/")
    for path in directories_made:
        for file in os.listdir(path):
            shutil.move(path + file, current_path + "volume_reduction/" + file)
        os.rmdir(path)
