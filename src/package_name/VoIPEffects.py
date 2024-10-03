#!/usr/bin/python

import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from package_name.utils import divide_list_randomly, make_directory, calculate_avg_sti

import random
from scipy.signal import resample
from package_name.sti import stiFromAudio

def calculate_STI(target_audio, reference_audio, refRate):
    try:
        STI = stiFromAudio(reference_audio, target_audio, refRate)
        print("STI: ", STI)
        return STI

    except Exception as e:
        print("Error in STI calculation:", e)
        return 1.0

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

def add_ambient_noise(audioPath, noisePath, snr_dB, sti_threshold):
    
    flag_fault = True

    while flag_fault:
        # Load the original audio file
        signal, sr = librosa.load(audioPath, sr=None)

        # Load the noise file
        noise_signal, noise_sr = librosa.load(noisePath, sr=None)

        # Resample the noise signal to match the sampling rate of the original signal
        noise_signal = librosa.resample(noise_signal, orig_sr=noise_sr, target_sr=sr)
        noise_sr = sr

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

        # Calculate the desired noise power based on the desired SNR
        snr_linear = 10 ** (snr_dB / 10.0)
        desired_noise_power = signal_power / snr_linear

        # Scale the noise signal to achieve the desired noise power
        scaled_noise = noise_signal * np.sqrt(desired_noise_power / noise_power)

        # Add the noise to the signal
        noisy_signal = signal + scaled_noise

        if sr <=16000:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=18000)
            noisy_signal = librosa.resample(noisy_signal, orig_sr=sr, target_sr=18000)
            sr = 18000

        # Calculate the STI of the noisy signal
        STI = stiFromAudio(signal, noisy_signal, sr)

        if STI > sti_threshold:
            flag_fault = False
            break
        else:
            snr_dB += 5
            flag_fault = True

    return noisy_signal, sr

def add_reverberation(audioPath:str, targetpath: str, selectable: int = 0, iir_path: str = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/iir.wav"):
    if selectable == 0:
        cmd1 = f"ffmpeg -loglevel error -i {audioPath} -map 0 -c:v copy -af aecho=1.0:0.9:70:0.5 {targetpath}"
        os.system(cmd1)
    elif selectable == 1:
        cmd2 = f"ffmpeg -loglevel error -i {audioPath} -i {iir_path} -filter_complex '[0] [1] afir=dry=10:wet=10 [reverb]; [0] [reverb] amix=inputs=2:weights=10 4' {targetpath}"
        os.system(cmd2)

    return



def add_voip_perterbation_effects(gaussian_SNR_levels: list, ambient_SNR_levels: list, ambient_noise_dir: str, reference_dir: str, sti_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mAdding VoIP Perterbation Effects\033[0m")
    print()

    flag_fault_0 = True
    gaussian_SNR_levels.sort(reverse=True)

    audio_files = os.listdir(reference_dir)

    os.chdir(reference_dir)
    os.chdir("../")

    print(os.getcwd())

    target_dir = os.getcwd() + "/augmented_data/VoIP_perterbations/"
    make_directory(target_dir)

    for audio in tqdm(audio_files, desc="Adding VoIP Perterbation Effects"):
        input_audio = reference_dir + str(audio)
        output_audio = target_dir + "voip_" + str(audio)

        # Append the output audio file to the list for text file creation
        output_files.append("voip_" + str(audio))

        # Add Reverberation Effects
        print("Adding Reverberation Effects")
        reverb_selectable = random.choice([0, 1])
        add_reverberation(input_audio, output_audio, selectable=reverb_selectable)

        #input_audio_signal, sr = librosa.load(input_audio, sr=None)

        # Add Gaussian Noise Effects
        """print("Adding Gaussian Noise Effects")
        desired_snr_dB = random.choice(gaussian_SNR_levels)

        while flag_fault_0:
            print("SNR: ", desired_snr_dB)
            gaussian_noise_signal, sample_rate = add_white_noise(input_audio, desired_snr_dB)
            sti = calculate_STI(gaussian_noise_signal, input_audio_signal, sample_rate)
            if sti < sti_threshold:
                print("STI is below the threshold. Trying another SNR level.")
                desired_snr_dB = gaussian_SNR_levels[gaussian_SNR_levels.index(desired_snr_dB) - 1]
                flag_fault_0 = True
            else:
                flag_fault_0 = False"""


        # Add Ambient Noise Effects
        """print("Adding Ambient Noise Effects")
        noise_files = os.listdir(ambient_noise_dir)
        desired_snr_dB = random.choice(ambient_SNR_levels)
        noise = random.choice(noise_files)
        noise_audio = ambient_noise_dir + str(noise)
        flag_fault_0 = True

        print("SNR: ", desired_snr_dB)
        ambient_noise_signal, sample_rate = add_ambient_noise(input_audio, noise_audio, desired_snr_dB, sti_threshold)"""

        




        # Save the output audio file
        #sf.write(output_audio, gaussian_noise_signal, sr)
    






    print()
    print("\033[92mPacket Loss Effect added successfully!\033[0m")

    # Create a text file to store the output audio files
    if len(output_files) > 0:
        os.chdir(reference_dir)
        os.chdir("../")

        with open('augmented_data/voip_perterbations.txt', 'w') as file:
            for item in output_files:
                file.write(f"{item}\n")