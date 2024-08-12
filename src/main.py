#!/usr/bin/python

# Imports
import os
import shutil
import random
import librosa
import argparse
import numpy as np
from pesq import pesq
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample
from simple_term_menu import TerminalMenu

def add_white_noise(audioPath, snr_dB):
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

def divide_list_randomly(lst, n):
    #Shuffle the list to ensure randomness
    random.shuffle(lst)
    
    #Determine the size of each part
    avg = len(lst) // n
    remainder = len(lst) % n

    #Create the parts
    parts = []
    start = 0

    for i in range(n):
        # Determine the end index for this part
        end = start + avg + (1 if i < remainder else 0)
        
        # Append the sliced part to the list of parts
        parts.append(lst[start:end])
        
        # Move the start index to the next section
        start = end

    return parts

def make_directory(directory):
    if not os.path.exists(directory):
        # Make a directory to store the augmented data
        os.makedirs(directory, exist_ok=True)
    else:
        print("Directory already exists. Confirm 'y' to overwrite the data.")
        confirm = input("Do you want to overwrite the data? (y/n): ")
        if confirm.lower() == "y":
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
    return

def calculate_avg_pesq(target_audio_list, target_dir, reference_dir, prefix = ""):

    for audio in tqdm(target_audio_list, desc="Calculating Average PESQ"):
        degrRate, target_Audio = wavfile.read(target_dir + prefix + str(audio))
        refRate, reference_audio = wavfile.read(reference_dir + str(audio))

        target_rate = 16000

        if degrRate != target_rate:
            number_of_samples = round(len(target_Audio) * float(target_rate) / degrRate)
            target_Audio = resample(target_Audio, number_of_samples)
            degrRate = target_rate

        if refRate != target_rate:
            number_of_samples = round(len(reference_audio) * float(target_rate) / refRate)
            reference_audio = resample(reference_audio, number_of_samples)
            refRate = target_rate

        try:
            PESQ = pesq(degrRate, reference_audio, target_Audio, 'wb')
            pesq_total += PESQ

        except Exception as e:
            print("Error in PESQ calculation:", e)
            pesq_total += 0.0
            continue
        
    return pesq_total/len(target_audio_list)

def main():
    # Define the menu options
    menu_options = [
        "Calculate Average PESQ",
        "Augment Data for Training",
        "Exit",
    ]

    # Create a TerminalMenu instance
    terminal_menu = TerminalMenu(menu_options, title="Main Menu", clear_screen=True)

    while True:
        # Show the menu and get the selected option
        selected_option_index = terminal_menu.show()

        # Perform actions based on the selected option
        if selected_option_index == 0:
            print(" "*50 + "Calculating Average PESQ" + " "*50)
            print()
            print("Target Directory: ", args.target_dir)
            print("Reference Directory: ", args.reference_dir)

            target_files = os.listdir(args.target_dir)
            reference_files = os.listdir(args.reference_dir)
            target_files.sort()
            reference_files.sort()

            pesq_total = 0.0

            for audio in tqdm(reference_files):
                degrRate, target_Audio = wavfile.read(args.target_dir + str(audio))
                refRate, reference_audio = wavfile.read(args.reference_dir + str(audio))

                target_rate = 16000

                if degrRate != target_rate:
                    number_of_samples = round(len(target_Audio) * float(target_rate) / degrRate)
                    target_Audio = resample(target_Audio, number_of_samples)
                    degrRate = target_rate

                if refRate != target_rate:
                    number_of_samples = round(len(reference_audio) * float(target_rate) / refRate)
                    reference_audio = resample(reference_audio, number_of_samples)
                    refRate = target_rate

                try:
                    PESQ = pesq(degrRate, reference_audio, target_Audio, 'wb')
                    pesq_total += PESQ

                except Exception as e:
                    print("Error in PESQ calculation:", e)
                    pesq_total += 0.0
                    continue
                
            print("Average PESQ: ", pesq_total/len(reference_files))
            

        elif selected_option_index == 1:
            output_files = []

            augment_data_menu_options = [
                "Add Gaussian Noise",
                "Add Ambient Noise",
                "Add Reverberation",
                "Add Muffling (Volume Reduction)",
                "Add Codec Losses",
                "Add Downsampling Effects",
                "Go Back"
            ]

            augment_data_menu = TerminalMenu(augment_data_menu_options, title="Augment Data Menu", clear_screen=True)
            augment_data_selected_option_index = augment_data_menu.show()

            if augment_data_selected_option_index == 0:
                print("\033[91mAdding Gaussian Noise\033[0m")
                print()

                SNR_levels_dB = [5, 10, 15, 20]
                SNR_levels_dB.sort(reverse=True)

                os.chdir(args.reference_dir)

                reference_files = os.listdir(args.reference_dir)

                audio_files_for_each_partition = len(reference_files) // len(SNR_levels_dB)
                print("Number of audio files for each SNR level: ", audio_files_for_each_partition)

                # Divide the list of audio files into partitions
                audio_files = divide_list_randomly(reference_files, len(SNR_levels_dB))

                # Check the existence of directory to store the augmented data exists
                for i in range(len(SNR_levels_dB)):
                    target_dir = "../augmented_data/gaussian_noise_" + str(SNR_levels_dB[i]) + "dB/"
                    make_directory(target_dir)
                    print("Directory created: ", target_dir)

                    # Add Gaussian Noise to the audio files
                    for audio in tqdm(audio_files[i], desc="Adding Gaussian Noise to Partition " + str(i+1)):
                        input_audio = args.reference_dir + str(audio)
                        output_audio = target_dir + "g" + str(SNR_levels_dB[i]) + "dB_" + str(audio)
                        desired_snr_dB = SNR_levels_dB[i]

                        output_files.append("g" + str(SNR_levels_dB[i]) + "dB_" + str(audio))

                        #print(input_audio, output_audio, desired_snr_dB)

                        noisy_signal, sample_rate = add_white_noise(input_audio, desired_snr_dB)

                        # Save the output with noise to a new file
                        sf.write(output_audio, noisy_signal, sample_rate)

                    print("Target Directory: ", args.target_dir)
                    print("Reference Directory: ", args.reference_dir)
                    avg_pesq = calculate_avg_pesq(audio_files[i], target_dir + "../augmented_data/", args.reference_dir, prefix = "g" + str(SNR_levels_dB[i]) + "dB_")

                    print("Average PESQ for SNR level ", SNR_levels_dB[i], "dB: ", avg_pesq)


            elif augment_data_selected_option_index == 1:
                print("Adding Ambient Noise")
            elif augment_data_selected_option_index == 2:
                print("Adding Reverberation")
            elif augment_data_selected_option_index == 3:
                print("Adding Muffling (Volume Reduction)")
            elif augment_data_selected_option_index == 4:
                print("Adding Codec Losses")
            elif augment_data_selected_option_index == 5:
                print("Adding Downsampling Effects")
            elif augment_data_selected_option_index == 6:
                continue

        elif selected_option_index == 2:
            print("Exiting the program. Goodbye!")
            break

        # Pause to allow the user to read the output
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--target_dir', type=str, help="path to the target audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/BaselineDataset/LA/ASVspoof2019_LA_eval/reverbEcho/")
    parser.add_argument('-r', '--reference_dir', type=str, help="path to the reference audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/BaselineDataset/LA/ASVspoof2019_LA_eval/original_wav/")
    args = parser.parse_args()
    main()