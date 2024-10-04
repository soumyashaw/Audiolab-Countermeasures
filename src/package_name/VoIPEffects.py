#!/usr/bin/python

import os
import random
import shutil
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from package_name.utils import make_directory
from package_name.sti import stiFromAudio, readwav


def calculate_STI(reference_audio, target_audio, ref_rate):

    if ref_rate <=16000:
        reference_audio = librosa.resample(reference_audio, orig_sr=ref_rate, target_sr=18000)
        target_audio = librosa.resample(target_audio, orig_sr=ref_rate, target_sr=18000)
        ref_rate = 18000

    try:
        STI = stiFromAudio(reference_audio, target_audio, ref_rate)
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

def add_ambient_noise(audioPath, noisePath, snr_dB, sti_threshold, reference_audio):
    
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
        STI = stiFromAudio(reference_audio, noisy_signal, sr)

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
        return encoder.apply(waveform, sr), sr
    
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

def simulate_packet_loss(input_audio, loss_rate):
    audio_data, sr = librosa.load(input_audio, sr=None)

    num_samples = len(audio_data)
    lost_samples = int(loss_rate * num_samples)
    indices_to_drop = np.random.choice(num_samples, lost_samples, replace=False)

    simulated_data = np.delete(audio_data, indices_to_drop)
    return simulated_data, sr

def find_volume(audio):
    rms = np.sqrt(np.mean(audio**2))

    # Convert the RMS value to decibels (dB)
    return 20 * np.log10(rms)



def add_voip_perterbation_effects(gaussian_SNR_levels: list, ambient_SNR_levels: list, ambient_noise_dir: str, volume_threshold: float, lower_sampling_rate: int, current_sampling_rate: int, packet_loss_rate: float, reference_dir: str, sti_threshold: float):
    output_files = []
    flag_fault = True

    print(" "*50 + "\033[91mAdding VoIP Perterbation Effects\033[0m")
    print()

    # List all the audio files in the reference directory (original audio files)
    audio_files = os.listdir(reference_dir)

    # Judge the amount of volume reduction justified using a random sample audio file
    vol_dB = 100.0
    dB_reduced = 1
    audio = random.choice(audio_files)
    reference_audio, sr = librosa.load(reference_dir + audio, sr=None)

    # Loop to find the amount of volume reduction justified
    while vol_dB > volume_threshold:
        db_reduction = -1 * dB_reduced
        reduction_factor = 10 ** (db_reduction / 20)

        volume_reduced_audio = reference_audio * reduction_factor

        vol_dB = find_volume(volume_reduced_audio)

        if vol_dB >= volume_threshold:
            dB_reduced += 1

    vol_dBs = [float(i) for i in range(1, dB_reduced)]

    os.chdir(reference_dir)
    os.chdir("../")

    target_dir = os.getcwd() + "/augmented_data/VoIP_perterbations/"
    make_directory(target_dir)

    for audio in tqdm(audio_files, desc="Adding VoIP Perterbation Effects"):
        flag_fault = True

        # Input Audio Loading Path
        input_audio = reference_dir + str(audio)

        # Load the input audio file
        input_audio_signal, sr  = readwav(input_audio)

        # ----- Start Reverberation Effects -----
        #print("Adding Reverberation Effects")

        # Output Audio Saving Path for Reverberation Effects
        output_audio = target_dir + "reve_" + str(audio)

        # Randomly select the reverb effect
        reverb_selectable = random.choice([0, 1])

        # Add the reverberation effect to the audio file
        add_reverberation(input_audio, output_audio, selectable=reverb_selectable)
        ### ----- End Reverberation Effects ----- 

        # Randomly select the background noise effect
        bg_noise_selection = random.choice([0, 1])

        if bg_noise_selection == 0:
            # ----- Start Gaussian Noise Effects ----- 
            #print("Adding Gaussian Noise Effects")

            # Sort the SNR levels in descending order
            gaussian_SNR_levels.sort(reverse=True)

            # Randomly select the SNR level for the Gaussian noise
            desired_snr_dB = random.choice(gaussian_SNR_levels)

            while flag_fault:

                # Add the Gaussian noise to the audio file
                gaussian_noise_signal, sample_rate = add_white_noise(target_dir + "reve_" + str(audio), desired_snr_dB)

                # Calculate the STI of the noisy signal
                sti = calculate_STI(input_audio_signal, gaussian_noise_signal, sample_rate)

                # Check if the STI is below the threshold
                if sti < sti_threshold:
                    desired_snr_dB += 2
                    flag_fault = True
                else:
                    flag_fault = False

            # Save the audio file with the Gaussian noise effect
            sf.write(target_dir + "bgno_" + str(audio), gaussian_noise_signal, sample_rate)
            # ----- End Gaussian Noise Effects  ----- 

        else:
            # ----- Start Ambient Noise Effects ----- 
            #print("Adding Ambient Noise Effects")

            # Enumerate the ambient noise files in the directory
            noise_files = os.listdir(ambient_noise_dir)

            # Randomly select the SNR level for the ambient noise
            desired_snr_dB = random.choice(ambient_SNR_levels)

            # Randomly choose the ambient noise file
            noise = random.choice(noise_files)

            noise_audio = ambient_noise_dir + str(noise)

            # Add the ambient noise to the audio file
            ambient_noise_signal, sample_rate = add_ambient_noise(target_dir + "reve_" + str(audio), noise_audio, desired_snr_dB, sti_threshold, input_audio_signal)

            # Save the audio file with the ambient noise effect
            sf.write(target_dir + "bgno_" + str(audio), ambient_noise_signal, sample_rate)
            # ----- End Ambient Noise Effects ----- 

        # ----- Start Volume Reduction Effects ----- 
        #print("Adding Volume Reduction Effects")

        # Load the reference audio file
        reference_audio, sr = librosa.load(target_dir + "bgno_" + str(audio), sr=None)

        # Randomly select the volume reduction level
        vol_dB = random.choice(vol_dBs)

        # Calculate the reduction factor based on the volume reduction level
        db_reduction = -1 * vol_dB
        reduction_factor = 10 ** (db_reduction / 20)
        volume_reduced_audio = reference_audio * reduction_factor

        # Save the audio file with the volume reduction effect
        sf.write(target_dir + "volu_" + str(audio), volume_reduced_audio, sr)

        # ----- End Volume Reduction Effects ----- 

        # ----- Start Codec Artifacts Effects ----- 
        #print("Adding Codec Artifacts Effects")

        # Apply G.722 codec for the audio file
        codec_added_audio, sample_rate = add_codec_loss(target_dir + "volu_" + str(audio), "wav", "g722")

        # Save the audio file with the codec artifacts effect
        sf.write(target_dir + "code_" + str(audio), codec_added_audio, sample_rate)
        # ----- End Codec Artifacts Effects ----- 

        # ----- Start Downsampling Effects ----- 
        #print("Adding Downsampling Effects")

        # Generate a list of sampling frequencies according to the lower and current sampling rates
        sampling_freqs = list(np.arange(lower_sampling_rate, current_sampling_rate, 5000))

        # Sort the sampling frequencies in descending order
        sampling_freqs.sort(reverse=True)

        # Randomly select the sampling frequency for the downsampling effect
        freq = random.choice(sampling_freqs)
        
        flag_fault = True
        while flag_fault:
            # Downsample the audio file
            downsampled_audio, sample_rate = downsample_audio(target_dir + "code_" + str(audio), freq, current_sampling_rate)
            sti = calculate_STI(input_audio_signal, downsampled_audio, sample_rate)
            if sti < sti_threshold:
                freq += 5000
                flag_fault = True
            else:
                flag_fault = False

        sf.write(target_dir + "down_" + str(audio), downsampled_audio, sr)
        # ----- End Downsampling Effects ----- 

        # ----- Start Packet Loss Effects ----- 
        #print("Adding Packet Loss Effects")
        loss_rate = packet_loss_rate
        packet_loss_audio, sample_rate = simulate_packet_loss(target_dir + "down_" + str(audio), loss_rate)
        # ----- End Packet Loss Effects ----- 


        # Append the output audio file to the list for text file creation
        output_files.append("voip_" + str(audio))

        output_audio = target_dir + "voip_" + str(audio)

        # Cleanup
        os.remove(target_dir + "reve_" + str(audio))
        os.remove(target_dir + "bgno_" + str(audio))
        os.remove(target_dir + "volu_" + str(audio))
        os.remove(target_dir + "code_" + str(audio))
        os.remove(target_dir + "down_" + str(audio))

        # Save the output audio file
        sf.write(output_audio, packet_loss_audio, sample_rate)
    

    print()
    print("\033[92mVoIP effects added successfully!\033[0m")

    # Create a text file to store the output audio files
    if len(output_files) > 0:
        os.chdir(reference_dir)
        os.chdir("../")

        with open('augmented_data/voip_perterbations.txt', 'w') as file:
            for item in output_files:
                file.write(f"{item}\n")