#!/usr/bin/python

# Imports
import os
import shutil
import random
import librosa
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.signal import resample
from package_name.sti import stiFromAudio, readwav
from simple_term_menu import TerminalMenu

from package_name.GaussianNoiseEffects import add_effects

def main():
    # Define the menu options
    menu_options = [
        "Calculate Average STI",
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
            print(" "*50 + "\033[91mCalculating Average STI\033[0m" + " "*50)
            print()
            print("Target Directory: ", args.target_dir)
            print("Reference Directory: ", args.reference_dir)

            target_files = os.listdir(args.target_dir)
            reference_files = os.listdir(args.reference_dir)
            target_files.sort()
            reference_files.sort()

            sti_total = 0.0

            for audio in tqdm(reference_files):
                target_Audio, degrRate  = readwav(args.target_dir + str(audio))
                reference_audio, refRate  = readwav(args.reference_dir + str(audio))

                try:
                    STI = stiFromAudio(reference_audio, target_Audio, refRate)
                    sti_total += STI

                except Exception as e:
                    print("Error in STI calculation:", e)
                    sti_total += 0.0
                    continue
                
            print("\033[91mAverage STI\033[0m: ", sti_total/len(reference_files))
            

        elif selected_option_index == 1:

            augment_data_menu_options = [
                "Add Gaussian Noise",
                "Add Ambient Noise",
                "Add Reverberation",
                "Add Muffling (Volume Reduction)",
                "Add Codec Artifacts",
                "Add Downsampling Effects",
                "Add Packet Loss Effects",
                "Go Back"
            ]

            augment_data_menu = TerminalMenu(augment_data_menu_options, title="Augment Data Menu", clear_screen=True)
            augment_data_selected_option_index = augment_data_menu.show()

            if augment_data_selected_option_index == 0:
                SNR_levels_dB = [5, 10, 15, 20, 25, 30]
                add_effects(SNR_levels_dB, args.reference_dir, args.sti_threshold)
            elif augment_data_selected_option_index == 1:
                output_files = []

                print(" "*50 + "\033[91mAdding Ambient Noise\033[0m")
                print()

                # Enumerate the ambient noise files in the directory
                noise_files = os.listdir(args.ambient_noise_dir)

                # Declare the SNR levels for the ambient noise
                SNR_levels_dB = [5, 10, 15, 20, 25, 30]

                # List all the audio files in the reference directory (original audio files)
                audio_files = os.listdir(args.reference_dir)

                # Change the directory to the reference directory
                os.chdir(args.reference_dir)

                os.chdir("../")

                # Make a directory to store the augmented data
                make_directory(os.getcwd() + "/augmented_data/ambient_noise/")

                target_dir = os.getcwd() + "/augmented_data/ambient_noise/"

                for audio in tqdm(audio_files, desc="Adding Ambient Noise to Audios"):
                    # Randomly choose the ambient noise file
                    noise = random.choice(noise_files)

                    # Randomly choose the SNR level
                    SNR = random.choice(SNR_levels_dB)

                    input_audio = args.reference_dir + str(audio)

                    noise_audio = args.ambient_noise_dir + str(noise)

                    # Append the identifier string to output audio file
                    output_audio = f"{target_dir}amb{str(SNR)}dB_{str(audio)}"

                    # Append the output audio file to the list for text file creation
                    output_files.append("amb" + str(SNR) + "dB_" + str(audio))

                    # Call the function to add white noise to the audio file
                    noisy_signal, sample_rate = add_ambient_noise(input_audio, noise_audio, SNR)

                    # Save the output with noise to a new file
                    sf.write(output_audio, noisy_signal, sample_rate)

                print()
                print("\033[92mAmbient Noise added successfully!\033[0m")

                # Create a text file to store the output audio files
                os.chdir(args.reference_dir)
                os.chdir("../")

                with open('augmented_data/ambient_noise.txt', 'w') as file:
                    for item in output_files:
                        file.write(f"{item}\n")

            elif augment_data_selected_option_index == 2:
                output_files = []

                print(" "*50 + "\033[91mAdding Reverberation\033[0m")
                print()

                # Collect the audio files from the reference directory
                audio_files = os.listdir(args.reference_dir)

                # Divide the audio files into two partitions for reverb effects
                audio_files = divide_list_randomly(audio_files, 2)

                # Change the directory to the reference directory
                os.chdir(args.reference_dir)

                os.chdir("../")

                # Make a directory to store the augmented data
                make_directory(os.getcwd() + "/augmented_data/reverberations/")

                for i in range(len(audio_files)):
                    for audio in tqdm(audio_files[i], desc="Adding Reverberation to Partition " + str(i+1)):
                        input_audio = args.reference_dir + str(audio)
                        output_audio = os.getcwd() + "/augmented_data/reverberations/" + "reverb_" + str(audio)

                        # Call the function to add reverb effects to the audio file
                        add_reverberation(input_audio, output_audio, i)

                        # Append the output audio file to the list for text file creation
                        output_files.append("reverb_" + str(audio))

                    avg_sti = calculate_avg_sti(audio_files[i], os.getcwd() + "/augmented_data/reverberations/", args.reference_dir, prefix = "reverb_")

                    # Print the average STI for the packet drop rate
                    print(f"Average STI for Reverberations Type {i+1}: {avg_sti}")

                    if avg_sti < args.sti_threshold:
                        print("\033[91mAverage STI is below the threshold.\033[0m Deleting augmented data.")

                        # Remove the directory made
                        shutil.rmtree(target_dir)

                        break

                print()
                print("\033[92mReverberations added successfully!\033[0m")

                # Create a text file to store the output audio files
                os.chdir(args.reference_dir)
                os.chdir("../")

                with open('augmented_data/reverberations.txt', 'w') as file:
                    for item in output_files:
                        file.write(f"{item}\n")

            elif augment_data_selected_option_index == 3:
                output_files = []

                print(" "*50 + "\033[91mAdding Muffling (Volume Reduction)\033[0m")
                print()

                audio_files = os.listdir(args.reference_dir)
                audio = random.choice(audio_files)

                # Load the audio file
                reference_audio, sr = librosa.load(args.reference_dir + audio, sr=None)

                target_rate = 16000

                # Modify sampling rate to 16kHz (for STI calculation)
                if sr != target_rate:
                    number_of_samples = round(len(reference_audio) * float(target_rate) / sr)
                    reference_audio = resample(reference_audio, number_of_samples)
                    sr = target_rate

                vol_dB = 100.0
                dB_reduced = 1

                while vol_dB > args.volume_threshold:
                    db_reduction = -1 * dB_reduced
                    reduction_factor = 10 ** (db_reduction / 20)

                    volume_reduced_audio = reference_audio * reduction_factor

                    vol_dB = find_volume(volume_reduced_audio)

                    if vol_dB >= args.volume_threshold:
                        dB_reduced += 1

                vol_dBs = [float(i) for i in range(1, dB_reduced)]
                
                directories_made = []

                # List all the audio files in the reference directory (original audio files)
                reference_files = os.listdir(args.reference_dir)

                # Divide the list of audio files into n partitions (based on the number of SNR levels)
                audio_files = divide_list_randomly(reference_files, len(vol_dBs))

                # Check the existence of directory to store the augmented data exists
                for i in range(len(vol_dBs)):
                    # Change the directory to the reference directory
                    os.chdir(args.reference_dir)

                    # Create a new directory to store the augmented data
                    os.chdir("../")
                    target_dir = os.getcwd() + "/augmented_data/vol_reduction_" + str(int(vol_dBs[i])) + "dB/"
                    make_directory(target_dir)
                    directories_made.append(target_dir)

                    # Reduce the volume of the audio files with given dB levels
                    for audio in tqdm(audio_files[i], desc="Reducing Volume in Partition " + str(i+1)):
                        input_audio = args.reference_dir + str(audio)
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
                os.chdir(args.reference_dir)
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

            elif augment_data_selected_option_index == 4:
                output_files = []

                print(" "*50 + "\033[91mAdding Codec Losses\033[0m")
                print()

                codecs = ['mulaw', 'g722', 'alaw', 'opus']

                # Empty list to store the Gaussian Noise directories with particular SNR levels
                directories_made = []
                
                # Change the directory to the reference directory
                os.chdir(args.reference_dir)

                # List all the audio files in the reference directory (original audio files)
                reference_files = os.listdir(args.reference_dir)

                # Divide the list of audio files into n partitions (based on the number of SNR levels)
                audio_files = divide_list_randomly(reference_files, len(codecs))

                # Check the existence of directory to store the augmented data exists
                for i in range(len(codecs)):
                    # Change the directory to the reference directory
                    os.chdir(args.reference_dir)

                    # Create a new directory to store the augmented data
                    os.chdir("../")
                    target_dir = os.getcwd() + "/augmented_data/" + str(codecs[i]) + "/"
                    make_directory(target_dir)
                    directories_made.append(target_dir)

                    # Add Gaussian Noise to the audio files with given SNR level
                    for audio in tqdm(audio_files[i], desc="Adding Codec Losses to Partition " + str(i+1)):
                        input_audio = args.reference_dir + str(audio)

                        # Append the identifier string to output audio file
                        output_audio = target_dir + str(codecs[i]) + "_" + str(audio)

                        # Append the output audio file to the list for text file creation
                        output_files.append(str(codecs[i]) + "_" + str(audio))

                        # Call the function to add Codec losses to the audio file
                        codec_added_audio = add_codec_loss(input_audio, "wav", codecs[i])

                        if codecs[i] == 'opus':
                            codec_added_audio.export(output_audio, format="wav")

                        else:
                            sf.write(output_audio, codec_added_audio, 16000)

                print()
                print("\033[92mCodec Artifacts added successfully!\033[0m")

                # Create a text file to store the output audio files
                os.chdir(args.reference_dir)
                os.chdir("../")

                with open('augmented_data/codec_losses.txt', 'w') as file:
                    for item in output_files:
                        file.write(f"{item}\n")

                # Cleanup: Merge the directories into one
                current_path = os.getcwd() + "/augmented_data/"
                make_directory(current_path + "codec_losses/")
                for path in directories_made:
                    for file in os.listdir(path):
                        shutil.move(path + file, current_path + "codec_losses/" + file)
                    os.rmdir(path)


            elif augment_data_selected_option_index == 5:
                output_files = []

                print(" "*50 + "\033[91mAdding Downsampling Effects\033[0m")
                print()

                flag_fault_5 = True

                # Target sampling frequencies for the downsampling effect
                sampling_freqs = list(np.arange(args.lower_sampling_rate, args.current_sampling_rate, 5000))

                # Sort the sampling frequencies in descending order
                sampling_freqs.sort(reverse=True)

                # Continue to augment the data until the STI threshold is met
                while flag_fault_5:
                
                    # Empty list to store the Downsampled directories with particular sampling frequencies
                    directories_made = []

                    # Change the directory to the reference directory
                    os.chdir(args.reference_dir)

                    # List all the audio files in the reference directory (original audio files)
                    reference_files = os.listdir(args.reference_dir)

                    # Divide the list of audio files into n partitions (based on the number of SNR levels)
                    audio_files = divide_list_randomly(reference_files, len(sampling_freqs))

                    # Check the existence of directory to store the augmented data exists
                    for i in range(len(sampling_freqs)):
                        # Change the directory to the reference directory
                        os.chdir(args.reference_dir)

                        # Create a new directory to store the augmented data
                        os.chdir("../")
                        target_dir = os.getcwd() + "/augmented_data/downsampling_" + str(int(sampling_freqs[i])) + "/"
                        make_directory(target_dir)
                        directories_made.append(target_dir)

                        # Reduce the volume of the audio files with given dB levels
                        for audio in tqdm(audio_files[i], desc="Downsampling audio in Partition " + str(i+1)):
                            input_audio = args.reference_dir + str(audio)
                            # Append the identifier string to output audio file
                            output_audio = target_dir + "ds_" + str(int(sampling_freqs[i])) + "_" + str(audio)

                            # Append the output audio file to the list for text file creation
                            output_files.append("ds_" + str(int(sampling_freqs[i])) + "_" + str(audio))

                            downsampled_audio, sr = downsample_audio(input_audio, sampling_freqs[i], args.current_sampling_rate)

                            sf.write(output_audio, downsampled_audio, sr)

                        # Calculate the average STI for the augmented data
                        target_dir = os.getcwd() + "/augmented_data/downsampling_" + str(int(sampling_freqs[i])) + "/"
                        avg_sti = calculate_avg_sti(audio_files[i], target_dir, args.reference_dir, prefix = "ds_" + str(int(sampling_freqs[i])) + "_")

                        # Print the average STI for the SNR level
                        print(f"Average STI for Sampling Frequency {sampling_freqs[i]} Hz: {avg_sti}")

                        # Check if the average STI is below the threshold
                        if avg_sti < args.sti_threshold:
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


            elif augment_data_selected_option_index == 6:
                output_files = []

                print(" "*50 + "\033[91mPacket Loss Effects\033[0m")
                print()

                audio_files = os.listdir(args.reference_dir)

                os.chdir(args.reference_dir)
                os.chdir("../")

                print(os.getcwd())

                target_dir = os.getcwd() + "/augmented_data/packet_loss/"
                make_directory(target_dir)

                for audio in tqdm(audio_files, desc="Adding Packet Loss Effects"):
                    input_audio = args.reference_dir + str(audio)
                    output_audio = target_dir + "pl_" + str(audio)

                    # Append the output audio file to the list for text file creation
                    output_files.append("pl_" + str(audio))

                    reference_audio, sr = librosa.load(input_audio, sr=None)

                    loss_rate = args.packet_loss_rate
                    packet_loss_audio = simulate_packet_loss(reference_audio, loss_rate)

                    sf.write(output_audio, packet_loss_audio, sr)

                avg_sti = calculate_avg_sti(audio_files, target_dir, args.reference_dir, prefix = "pl_")

                # Print the average STI for the packet drop rate
                print(f"Average STI for {loss_rate} packet drop rate: {avg_sti}")

                if avg_sti < args.sti_threshold:
                    print("\033[91mAverage STI is below the threshold.\033[0m Deleting augmented data.")

                    # Remove the directory made
                    shutil.rmtree(target_dir)

                print()
                print("\033[92mPacket Loss Effect added successfully!\033[0m")

                # Create a text file to store the output audio files
                os.chdir(args.reference_dir)
                os.chdir("../")

                with open('augmented_data/packet_loss.txt', 'w') as file:
                    for item in output_files:
                        file.write(f"{item}\n")

            elif augment_data_selected_option_index == 6:
                continue

        elif selected_option_index == 2:
            print("Terminating Execution. Goodbye!")
            break

        # Pause to allow the user to read the output
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--target_dir', type=str, help="path to the target audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/BaselineDataset/LA/ASVspoof2019_LA_eval/reverbEcho/")
    parser.add_argument('-r', '--reference_dir', type=str, help="path to the reference audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TrialData/OriginalData/")
    parser.add_argument('-s', '--sti_threshold', type=float, help="STI threshold for the augmented data", default=0.75)
    parser.add_argument('-v', '--volume_threshold', type=float, help="Volume threshold for the augmented data", default=-35)
    parser.add_argument('-l', '--packet_loss_rate', type=float, help="Target Packet Loss Rate for the augmented data", default=0.1)
    parser.add_argument('-m', '--lower_sampling_rate', type=int, help="Lower bound sampling rate to be applied to the audios", default=3400)
    parser.add_argument('-e', '--current_sampling_rate', type=int, help="Current sampling rate of the audio files", default=16000)
    parser.add_argument('-i', '--input_format', type=str, help="Input format of the audio files", default="wav")
    parser.add_argument('-o', '--output_format', type=str, help="Output format of the audio files", default="wav")
    parser.add_argument('-n', '--ambient_noise_dir', type=str, help="path to the ambient noise files to be used", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/ambient_noise/")
    args = parser.parse_args()
    main()