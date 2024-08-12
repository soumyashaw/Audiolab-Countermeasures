#!/usr/bin/python

# Imports
import os
import shutil
import librosa
import argparse
import numpy as np
from pesq import pesq
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample
from simple_term_menu import TerminalMenu

def addWhiteNoise(audioPath, snr_dB):
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

            counter = 0
            pesq_total = 0.0

            for audio in tqdm(reference_files):
                degrRate, target_Audio = wavfile.read(args.target_dir + str(audio))
                refRate, reference_audio = wavfile.read(args.reference_dir + str(audio))

                target_rate = 16000

                number_of_samples = round(len(target_Audio) * float(target_rate) / degrRate)
                target_Audio = resample(target_Audio, number_of_samples)
                degrRate = target_rate

                number_of_samples = round(len(reference_audio) * float(target_rate) / refRate)
                reference_audio = resample(reference_audio, number_of_samples)
                refRate = target_rate

                counter += 1
                try:
                    PESQ = pesq(degrRate, reference_audio, target_Audio, 'wb')
                    pesq_total += PESQ
                    #print(counter, " ", PESQ)

                except Exception as e:
                    print("Error in PESQ calculation:", e)
                    pesq_total += 0.0
                    continue
                
            print("Average PESQ: ", pesq_total/len(reference_files))
            

        elif selected_option_index == 1:
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

                reference_files = os.listdir(args.reference_dir)
                reference_files.sort()

                audio_files_for_each_partition = len(reference_files) // len(SNR_levels_dB)
                print("Number of audio files for each SNR level: ", audio_files_for_each_partition)

                # Check the existence of directory to store the augmented data exists
                if not os.path.exists("augmented_data/gaussian_noise"):
                    # Make a directory to store the augmented data
                    os.makedirs("augmented_data/gaussian_noise", exist_ok=True)
                else:
                    print("Directory already exists. Confirm 'y' to overwrite the data.")
                    confirm = input("Do you want to overwrite the data? (y/n): ")
                    if confirm.lower() == "y":
                        shutil.rmtree('augmented_data/gaussian_noise')
                        os.makedirs("augmented_data/gaussian_noise", exist_ok=True)
                    else:
                        continue
                
                # Add Gaussian Noise to the audio files
                for audio in audioFiles:
                    input_audio = parent_dir + str(audio)
                    output_audio = target_dir + str(audio)
                    desired_snr_dB = 20

                    noisy_signal, sample_rate = addWhiteNoise(input_audio, desired_snr_dB)

                    # Save the output with noise to a new file
                    sf.write(output_audio, noisy_signal, sample_rate)




                SNR_levels_dB.sort(reverse=True)
                for SNR in SNR_levels_dB:
                    print("SNR", SNR)

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