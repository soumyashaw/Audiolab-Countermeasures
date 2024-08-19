#!/usr/bin/python

# Imports
import os
import shutil
import random
import librosa
import argparse
import torchaudio
import numpy as np
from pesq import pesq
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.signal import resample
from simple_term_menu import TerminalMenu
#from torchmetrics.audio import PerceptualEvaluationSpeechQuality

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
    #pesq_total2 = 0.0

    # Test
    #wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

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

            # Test
            #PESQ2 = wb_pesq(reference_audio, target_Audio)
            #pesq_total2 += PESQ2

        except Exception as e:
            print("Error in PESQ calculation:", e)
            pesq_total += 0.0
            #pesq_total2 += 0.0
            continue
        
    #print("PESQ2: ", pesq_total2/len(target_audio_list))
    return pesq_total/len(target_audio_list)

def find_volume(audio):
    rms = np.sqrt(np.mean(audio**2))

    # Convert the RMS value to decibels (dB)
    return 20 * np.log10(rms)

def simulate_packet_loss(audio_data, loss_rate):
    num_samples = len(audio_data)
    lost_samples = int(loss_rate * num_samples)
    indices_to_drop = np.random.choice(num_samples, lost_samples, replace=False)

    simulated_data = np.delete(audio_data, indices_to_drop)
    return simulated_data

def add_codec_loss(audioPath, format, codec: str):
    if codec == 'mulaw' or codec == 'alaw' or codec == 'g722':
        # Load the audio file
        waveform, sr = torchaudio.load(audioPath, channels_first=False)

        # Assign Encoder based on the codec
        if codec == 'mulaw':
            encoder = "pcm_mulaw"
        elif codec == 'alaw':
            encoder = "pcm_alaw"
        elif codec == 'g722':
            encoder = "g722"

        # Apply the codec to the audio file
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)

        # Return the audio file with the codec applied
        return encoder.apply(waveform, sr)
    
    elif codec == 'opus':
        # Load the audio file
        audio = AudioSegment.from_file(audioPath)

        # Export as Opus format
        audio.export('encoded.opus', format="opus")

        # Load the encoded audio file
        audio = AudioSegment.from_file('encoded.opus')

        try:
            os.remove('encoded.opus')
        except:
            pass

        return audio
    
def add_reverberation(audioPath:str, targetpath: str, selectable: int = 0, iir_path: str = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/iir.wav"):
    if selectable == 0:
        cmd1 = f"ffmpeg -loglevel error -i {audioPath} -map 0 -c:v copy -af aecho=1.0:0.9:70:0.5 {targetpath}"
        os.system(cmd1)
    elif selectable == 1:
        cmd2 = f"ffmpeg -loglevel error -i {audioPath} -i {iir_path} -filter_complex '[0] [1] afir=dry=10:wet=10 [reverb]; [0] [reverb] amix=inputs=2:weights=10 4' {targetpath}"
        os.system(cmd2)

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

def add_ambient_noise(audioPath, noisePath, snr_dB):
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

        target_rate = 16000

        if sr != target_rate:
            number_of_samples = round(len(signal) * float(target_rate) / sr)
            temp_signal = resample(signal, number_of_samples)
            temp_noisy_signal = resample(noisy_signal, number_of_samples)
            temp_sr = target_rate

            # Calculate the PESQ of the noisy signal
            PESQ = pesq(temp_sr, temp_signal, temp_noisy_signal, 'wb')

        else:
            # Calculate the PESQ of the noisy signal
            PESQ = pesq(sr, signal, noisy_signal, 'wb')
        

        if PESQ > args.pesq_threshold:
            flag_fault = False
        else:
            snr_dB += 5
            flag_fault = True

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
                output_files = []

                print(" "*50 + "\033[91mAdding Gaussian Noise\033[0m")
                print()

                flag_fault_0 = True
                SNR_levels_dB = [5, 10, 15, 20, 25, 30]
                SNR_levels_dB.sort(reverse=True)

                # Continue to augment the data until the PESQ threshold is met
                while flag_fault_0:
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
                            flag_fault_0 = True
                        else:
                            # Set the flag to False to stop augmenting the data
                            flag_fault_0 = False

                        print()

                print("\033[92mGaussian Noise added successfully!\033[0m")

                # Create a text file to store the output audio files
                os.chdir(args.reference_dir)
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

                    avg_pesq = calculate_avg_pesq(audio_files[i], os.getcwd() + "/augmented_data/reverberations/", args.reference_dir, prefix = "reverb_")

                    # Print the average PESQ for the packet drop rate
                    print(f"Average PESQ for Reverberations Type {i+1}: {avg_pesq}")

                    if avg_pesq < args.pesq_threshold:
                        print("\033[91mAverage PESQ is below the threshold.\033[0m Deleting augmented data.")

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

                # Modify sampling rate to 16kHz (for PESQ calculation)
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

                # Continue to augment the data until the PESQ threshold is met
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

                        # Calculate the average PESQ for the augmented data
                        target_dir = os.getcwd() + "/augmented_data/downsampling_" + str(int(sampling_freqs[i])) + "/"
                        avg_pesq = calculate_avg_pesq(audio_files[i], target_dir, args.reference_dir, prefix = "ds_" + str(int(sampling_freqs[i])) + "_")

                        # Print the average PESQ for the SNR level
                        print(f"Average PESQ for Sampling Frequency {sampling_freqs[i]} Hz: {avg_pesq}")

                        # Check if the average PESQ is below the threshold
                        if avg_pesq < args.pesq_threshold:
                            # Remove the SNR level from the list
                            print("\033[91mAverage PESQ is below the threshold.\033[0m Augmenting with modified Downsampling levels.")
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

                avg_pesq = calculate_avg_pesq(audio_files, target_dir, args.reference_dir, prefix = "pl_")

                # Print the average PESQ for the packet drop rate
                print(f"Average PESQ for {loss_rate} packet drop rate: {avg_pesq}")

                if avg_pesq < args.pesq_threshold:
                    print("\033[91mAverage PESQ is below the threshold.\033[0m Deleting augmented data.")

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
    parser.add_argument('-p', '--pesq_threshold', type=float, help="PESQ threshold for the augmented data", default=1.0)
    parser.add_argument('-v', '--volume_threshold', type=float, help="Volume threshold for the augmented data", default=-35)
    parser.add_argument('-l', '--packet_loss_rate', type=float, help="Target Packet Loss Rate for the augmented data", default=0.1)
    parser.add_argument('-s', '--lower_sampling_rate', type=int, help="Lower bound sampling rate to be applied to the audios", default=3400)
    parser.add_argument('-e', '--current_sampling_rate', type=int, help="Current sampling rate of the audio files", default=16000)
    parser.add_argument('-n', '--ambient_noise_dir', type=str, help="path to the ambient noise files to be used", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/ambient_noise/")
    args = parser.parse_args()
    main()