#!/usr/bin/python

import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from package_name.utils import divide_list_randomly, make_directory, calculate_avg_sti

def add_white_noise(audioPath: str, snr_dB):
    # Load the audio file
    signal, sr = librosa.load(audioPath, sr=None)

    # Calculate the power of the signal
    signal_power = np.sum(signal ** 2) / len(signal)

    # Calculate the noise power based on the desired SNR
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise with the same length as the signal
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(signal))

    # Add the noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, sr

def add_gaussian_noise_effects(SNR_levels_dB: list, reference_dir: str, sti_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mAdding Gaussian Noise\033[0m")
    print()

    flag_fault_0 = True
    SNR_levels_dB.sort(reverse=True)

    # Continue to augment the data until the STI threshold is met
    while flag_fault_0:
        # Empty list to store the Gaussian Noise directories with particular SNR levels
        directories_made = []
        
        # Change the directory to the reference directory
        os.chdir(reference_dir)

        # List all the audio files in the reference directory (original audio files)
        reference_files = os.listdir(reference_dir)

        # Divide the list of audio files into n partitions (based on the number of SNR levels)
        audio_files = divide_list_randomly(reference_files, len(SNR_levels_dB))

        # Check the existence of directory to store the augmented data exists
        for i in range(len(SNR_levels_dB)):
            # Set the flag to True initially
            flag_fault_0 = True

            # Change the directory to the reference directory
            os.chdir(reference_dir)

            # Create a new directory to store the augmented data
            os.chdir("../")
            target_dir = os.getcwd() + "/augmented_data/gaussian_noise_" + str(SNR_levels_dB[i]) + "dB/"
            make_directory(target_dir)
            directories_made.append(target_dir)

            # Add Gaussian Noise to the audio files with given SNR level
            for audio in tqdm(audio_files[i], desc="Adding Gaussian Noise to Partition " + str(i+1)):
                input_audio = reference_dir + str(audio)
                # Append the identifier string to output audio file
                output_audio = target_dir + "g" + str(SNR_levels_dB[i]) + "dB_" + str(audio)
                desired_snr_dB = SNR_levels_dB[i]

                # Append the output audio file to the list for text file creation
                output_files.append("g" + str(SNR_levels_dB[i]) + "dB_" + str(audio))

                # Call the function to add white noise to the audio file
                noisy_signal, sample_rate = add_white_noise(input_audio, desired_snr_dB)

                # Save the output with noise to a new file
                sf.write(output_audio, noisy_signal, sample_rate)

            # Calculate the average STI for the augmented data
            target_dir = os.getcwd() + "/augmented_data/gaussian_noise_" + str(SNR_levels_dB[i]) + "dB/"
            avg_sti = calculate_avg_sti(audio_files[i], target_dir, reference_dir, prefix = "g" + str(SNR_levels_dB[i]) + "dB_")

            # Print the average STI for the SNR level
            print("Average STI for SNR level ", SNR_levels_dB[i], "dB: ", avg_sti)

            print("1:flag_fault_0: ", flag_fault_0)

            # Check if the average STI is below the threshold
            if avg_sti < sti_threshold:
                # Remove the SNR level from the list
                print("\033[91mAverage STI is below the threshold.\033[0m Augmenting with modified SNR levels.")
                SNR_levels_dB.pop()

                # Remove the directories made
                for path in directories_made:
                    shutil.rmtree(path)

                # Set the flag to True to continue augmenting the data
                flag_fault_0 = True
                break
            else:
                # Set the flag to False to stop augmenting the data
                flag_fault_0 = False

            print("2:flag_fault_0: ", flag_fault_0)
            _ = input("Press any key to continue...")
        print("SNR Levels: ",SNR_levels_dB)

    print("\033[92mGaussian Noise added successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(reference_dir)
    os.chdir("../")

    with open('augmented_data/gaussian_noise.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")

    # Cleanup: Merge the directories into one
    current_path = os.getcwd() + "/augmented_data/"
    make_directory(current_path + "gaussian_noise/")
    for path in directories_made:
        for file in os.listdir(path):
            shutil.move(path + file, current_path + "gaussian_noise/" + file)
        os.rmdir(path)
