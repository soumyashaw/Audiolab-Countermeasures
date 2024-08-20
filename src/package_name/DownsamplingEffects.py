#!/usr/bin/python

import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
from utils import divide_list_randomly, make_directory, calculate_avg_sti

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

def add_downsampling_effects(reference_dir: str, lower_sampling_rate: int, current_sampling_rate:int, sti_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mAdding Downsampling Effects\033[0m")
    print()

    flag_fault_5 = True

    # Target sampling frequencies for the downsampling effect
    sampling_freqs = list(np.arange(lower_sampling_rate, current_sampling_rate, 5000))

    # Sort the sampling frequencies in descending order
    sampling_freqs.sort(reverse=True)

    # Continue to augment the data until the STI threshold is met
    while flag_fault_5:
    
        # Empty list to store the Downsampled directories with particular sampling frequencies
        directories_made = []

        # Change the directory to the reference directory
        os.chdir(reference_dir)

        # List all the audio files in the reference directory (original audio files)
        reference_files = os.listdir(reference_dir)

        # Divide the list of audio files into n partitions (based on the number of SNR levels)
        audio_files = divide_list_randomly(reference_files, len(sampling_freqs))

        # Check the existence of directory to store the augmented data exists
        for i in range(len(sampling_freqs)):
            # Change the directory to the reference directory
            os.chdir(reference_dir)

            # Create a new directory to store the augmented data
            os.chdir("../")
            target_dir = os.getcwd() + "/augmented_data/downsampling_" + str(int(sampling_freqs[i])) + "/"
            make_directory(target_dir)
            directories_made.append(target_dir)

            # Reduce the volume of the audio files with given dB levels
            for audio in tqdm(audio_files[i], desc="Downsampling audio in Partition " + str(i+1)):
                input_audio = reference_dir + str(audio)
                # Append the identifier string to output audio file
                output_audio = target_dir + "ds_" + str(int(sampling_freqs[i])) + "_" + str(audio)

                # Append the output audio file to the list for text file creation
                output_files.append("ds_" + str(int(sampling_freqs[i])) + "_" + str(audio))

                downsampled_audio, sr = downsample_audio(input_audio, sampling_freqs[i], current_sampling_rate)

                sf.write(output_audio, downsampled_audio, sr)

            # Calculate the average STI for the augmented data
            target_dir = os.getcwd() + "/augmented_data/downsampling_" + str(int(sampling_freqs[i])) + "/"
            avg_sti = calculate_avg_sti(audio_files[i], target_dir, reference_dir, prefix = "ds_" + str(int(sampling_freqs[i])) + "_")

            # Print the average STI for the SNR level
            print(f"Average STI for Sampling Frequency {sampling_freqs[i]} Hz: {avg_sti}")

            # Check if the average STI is below the threshold
            if avg_sti < sti_threshold:
                # Remove the SNR level from the list
                print("\033[91mAverage STI is below the threshold.\033[0m Augmenting with modified Downsampling levels.")
                SNR_levels_dB.pop()

                # Remove the directories made
                for path in directories_made:
                    shutil.rmtree(path)

                # Set the flag to True to continue augmenting the data
                flag_fault_5 = True
            else:
                # Set the flag to False to stop augmenting the data
                flag_fault_5 = False
            print()

    print("\033[92mDownsampled the audios successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(args.reference_dir)
    os.chdir("../")

    with open('augmented_data/downsampling.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")

    # Cleanup: Merge the directories into one
    current_path = os.getcwd() + "/augmented_data/"
    make_directory(current_path + "downsampling/")
    for path in directories_made:
        for file in os.listdir(path):
            shutil.move(path + file, current_path + "downsampling/" + file)
        os.rmdir(path)