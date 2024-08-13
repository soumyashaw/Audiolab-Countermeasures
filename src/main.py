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

    pesq_total = 0.0

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
            print(" "*50 + "\033[91mCalculating Average PESQ\033[0m" + " "*50)
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
                
            print("\033[91mAverage PESQ\033[0m: ", pesq_total/len(reference_files))
            

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
                print(" "*50 + "\033[91mAdding Gaussian Noise\033[0m")
                print()

                flag_fault = True
                SNR_levels_dB = [5, 10, 15, 20]
                SNR_levels_dB.sort(reverse=True)

                # Continue to augment the data until the PESQ threshold is met
                while flag_fault:
                    # Empty list to store the Gaussian Noise directories with particular SNR levels
                    directories_made = []
                    
                    # Change the directory to the reference directory
                    os.chdir(args.reference_dir)

                    # List all the audio files in the reference directory (original audio files)
                    reference_files = os.listdir(args.reference_dir)

                    # Divide the list of audio files into n partitions (based on the number of SNR levels)
                    audio_files = divide_list_randomly(reference_files, len(SNR_levels_dB))

                    # Check the existence of directory to store the augmented data exists
                    for i in range(len(SNR_levels_dB)):
                        # Change the directory to the reference directory
                        os.chdir(args.reference_dir)

                        # Create a new directory to store the augmented data
                        os.chdir("../")
                        target_dir = os.getcwd() + "/augmented_data/gaussian_noise_" + str(SNR_levels_dB[i]) + "dB/"
                        make_directory(target_dir)
                        directories_made.append(target_dir)

                        # Add Gaussian Noise to the audio files with given SNR level
                        for audio in tqdm(audio_files[i], desc="Adding Gaussian Noise to Partition " + str(i+1)):
                            input_audio = args.reference_dir + str(audio)
                            # Append the identifier string to output audio file
                            output_audio = target_dir + "g" + str(SNR_levels_dB[i]) + "dB_" + str(audio)
                            desired_snr_dB = SNR_levels_dB[i]

                            # Append the output audio file to the list for text file creation
                            output_files.append("g" + str(SNR_levels_dB[i]) + "dB_" + str(audio))

                            # Call the function to add white noise to the audio file
                            noisy_signal, sample_rate = add_white_noise(input_audio, desired_snr_dB)

                            # Save the output with noise to a new file
                            sf.write(output_audio, noisy_signal, sample_rate)

                        # Calculate the average PESQ for the augmented data
                        target_dir = os.getcwd() + "/augmented_data/gaussian_noise_" + str(SNR_levels_dB[i]) + "dB/"
                        avg_pesq = calculate_avg_pesq(audio_files[i], target_dir, args.reference_dir, prefix = "g" + str(SNR_levels_dB[i]) + "dB_")

                        # Print the average PESQ for the SNR level
                        print("Average PESQ for SNR level ", SNR_levels_dB[i], "dB: ", avg_pesq)

                        # Check if the average PESQ is below the threshold
                        if avg_pesq < args.pesq_threshold:
                            # Remove the SNR level from the list
                            print("\033[91mAverage PESQ is below the threshold.\033[0m Augmenting with modified SNR levels.")
                            SNR_levels_dB.pop()

                            # Remove the directories made
                            for path in directories_made:
                                shutil.rmtree(path)

                            # Set the flag to True to continue augmenting the data
                            flag_fault = True
                        else:
                            # Set the flag to False to stop augmenting the data
                            flag_fault = False

                        print()

                print("\033[92mGaussian Noise added successfully!\033[0m")

                # Cleanup: Merge the directories into one
                current_path = os.getcwd() + "/augmented_data/"
                make_directory(current_path + "gaussian_noise/")
                for path in directories_made:
                    for file in os.listdir(path):
                        shutil.move(path + file, current_path + "gaussian_noise/" + file)
                    os.rmdir(path)

            elif augment_data_selected_option_index == 1:
                print(" "*50 + "\033[91mAdding Ambient Noise\033[0m")
                print()

            elif augment_data_selected_option_index == 2:
                print(" "*50 + "\033[91mAdding Reverberation\033[0m")
                print()

            elif augment_data_selected_option_index == 3:
                print(" "*50 + "\033[91mAdding Muffling (Volume Reduction)\033[0m")
                print()

                audio_files = os.listdir(args.reference_dir)
                audio = random.choice(audio_files)
                print(audio)

                # Load the audio file
                reference_audio, sr = librosa.load(args.reference_dir + audio, sr=None)

                target_rate = 16000

                # Modify sampling rate to 16kHz (for PESQ calculation)
                if sr != target_rate:
                    number_of_samples = round(len(reference_audio) * float(target_rate) / sr)
                    reference_audio = resample(reference_audio, number_of_samples)
                    sr = target_rate



                db_reduction = -1
                reduction_factor = 10 ** (db_reduction / 20)

                volume_reduced_audio = reference_audio * reduction_factor

                PESQ = pesq(sr, reference_audio, volume_reduced_audio, 'wb')
                print("PESQ(-1): ", PESQ)


                db_reduction = -10
                reduction_factor = 10 ** (db_reduction / 20)

                volume_reduced_audio = reference_audio * reduction_factor

                PESQ = pesq(sr, reference_audio, volume_reduced_audio, 'wb')
                print("PESQ(-10): ", PESQ)


                PESQ = 5.0
                dB_reduced = 1

                while PESQ > args.pesq_threshold:
                    db_reduction = -1 * dB_reduced
                    reduction_factor = 10 ** (db_reduction / 20)

                    volume_reduced_audio = reference_audio * reduction_factor

                    PESQ = pesq(sr, reference_audio, volume_reduced_audio, 'wb')
                    print("PESQ: ", PESQ, dB_reduced)

                    if PESQ >= args.pesq_threshold:
                        dB_reduced += 1

                



            elif augment_data_selected_option_index == 4:
                print(" "*50 + "\033[91mAdding Codec Losses\033[0m")
                print()

            elif augment_data_selected_option_index == 5:
                print(" "*50 + "\033[91mAdding Downsampling Effects\033[0m")
                print()

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
    parser.add_argument('-r', '--reference_dir', type=str, help="path to the reference audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TrialData/OriginalData/")
    parser.add_argument('-p', '--pesq_threshold', type=float, help="PESQ threshold for the augmented data", default=1.0)
    args = parser.parse_args()
    main()