import os
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
#from package_name.sti import stiFromAudio

def make_directory(directory, ignore=False):
    if not os.path.exists(directory):
        # Make a directory to store the augmented data
        os.makedirs(directory, exist_ok=True)
    else:
        if ignore:
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        else:
            print("Directory already exists. Confirm 'y' to overwrite the data.")
            confirm = input("Do you want to overwrite the data? (y/n): ")
            if confirm.lower() == "y":
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
            elif confirm.lower() == "n":
                print("Exiting the program.")
                exit(0)
            else:  
                print("Invalid input. Exiting the program.")
                exit(0)
    return


def add_ambient_noise(audioPath, noisePath, snr_dB, sti_threshold):

    flag_fault = True
    STI = 0.57

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

        print("Signal Shape: ", signal.shape)
        print("Scaled Noise Shape: ", scaled_noise.shape)

        # Add the noise to the signal
        noisy_signal = signal + scaled_noise

        if sr <=16000:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=18000)
            noisy_signal = librosa.resample(noisy_signal, orig_sr=sr, target_sr=18000)
            sr = 18000

        # Calculate the STI of the noisy signal
        """STI = stiFromAudio(signal, noisy_signal, sr)
        print("STI: ", STI)"""
        #STI = 0.57

        if STI > sti_threshold:
            flag_fault = False
            break
        else:
            snr_dB += 5
            STI = 0.62
            flag_fault = True

        print("Flag Fault: ", flag_fault)

    return noisy_signal, sr

def add_ambient_noise_effects(SNR_levels_dB, reference_dir, ambient_noise_dir, sti_threshold):
    output_files = []

    print(" "*50 + "\033[91mAdding Ambient Noise\033[0m")
    print()

    # Enumerate the ambient noise files in the directory
    noise_files = os.listdir(ambient_noise_dir)

    # Declare the SNR levels for the ambient noise
    

    # List all the audio files in the reference directory (original audio files)
    audio_files = os.listdir(reference_dir)

    # Change the directory to the reference directory
    os.chdir(reference_dir)

    os.chdir("../")

    # Make a directory to store the augmented data
    make_directory(os.getcwd() + "/augmented_data/ambient_noise/")

    target_dir = os.getcwd() + "/augmented_data/ambient_noise/"

    for audio in tqdm(audio_files, desc="Adding Ambient Noise to Audios"):
        audio = "LA_T_9916745.wav"

        # Randomly choose the ambient noise file
        noise = random.choice(noise_files)

        # Randomly choose the SNR level
        SNR = random.choice(SNR_levels_dB)

        input_audio = reference_dir + str(audio)

        noise_audio = ambient_noise_dir + str(noise)

        # Append the identifier string to output audio file
        output_audio = f"{target_dir}amb{str(SNR)}dB_{str(audio)}"

        print("Audio: ", audio)

        # Append the output audio file to the list for text file creation
        output_files.append("amb" + str(SNR) + "dB_" + str(audio))

        # Call the function to add white noise to the audio file
        noisy_signal, sample_rate = add_ambient_noise(input_audio, noise_audio, SNR, sti_threshold)

        # Save the output with noise to a new file
        sf.write(output_audio, noisy_signal, sample_rate)
        print("Output Audio Saved")

    print()
    print("\033[92mAmbient Noise added successfully!\033[0m")

    if len(output_files) > 0:
        # Create a text file to store the output audio files
        os.chdir(reference_dir)
        os.chdir("../")

        with open('augmented_data/ambient_noise.txt', 'w') as file:
            for item in output_files:
                file.write(f"{item}\n")


SNR_levels_dB = [5, 10, 15, 20, 25, 30]
ref_dir = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/ASVspoof2019/LA/ASVspoof2019_LA_train/wav/"
amb_noise_dir = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/ambient_noise/"
sti_threshold = 0.6
add_ambient_noise_effects(SNR_levels_dB, ref_dir, amb_noise_dir, sti_threshold)






### Dev set
import os

filename = "original_ASVspoof2019.LA.cm.dev.trl.txt"
dict = {}

with open(filename, "r") as file1:
    for line in file1:
        line = line.strip().split()
        dict[line[1]] = line[4]
        
loc1 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/ambient_noise/'
loc2 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/codec_losses/'
loc3 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/downsampling/'
loc4 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/gaussian_noise/'
loc5 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/packet_loss/'
loc6 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/reverberations/'
loc7 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/volume_reduction/'
loc8 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac/'

loc1_files = os.listdir(loc1)
loc2_files = os.listdir(loc2)
loc3_files = os.listdir(loc3)
loc4_files = os.listdir(loc4)
loc5_files = os.listdir(loc5)
loc6_files = os.listdir(loc6)
loc7_files = os.listdir(loc7)
loc8_files = os.listdir(loc8)

dict1 = {}
for file in loc1_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict1[original_file] = dict[file]

dict2 = {}
for file in loc2_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict2[original_file] = dict[file]

dict3 = {}
for file in loc3_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict3[original_file] = dict[file]

dict4 = {}
for file in loc4_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict4[original_file] = dict[file]

dict5 = {}
for file in loc5_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict5[original_file] = dict[file]

dict6 = {}
for file in loc6_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict6[original_file] = dict[file]

dict7 = {}
for file in loc7_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict7[original_file] = dict[file]

dict8 = {}
for file in loc8_files:
    original_file = file[:-5]
    file = file[:-5]
    dict8[original_file] = dict[file]


with open("protocol.txt", "w") as file2:
    
    key1 = list(dict1.keys())
    for k1 in key1:
        file2.write(f"LA_0000 {k1} - - {dict1[k1]}\n")

    key2 = list(dict2.keys())
    for k2 in key2:
        file2.write(f"LA_0000 {k2} - - {dict2[k2]}\n")

    key3 = list(dict3.keys())
    for k3 in key3:
        file2.write(f"LA_0000 {k3} - - {dict3[k3]}\n")

    key4 = list(dict4.keys())
    for k4 in key4:
        file2.write(f"LA_0000 {k4} - - {dict4[k4]}\n")

    key5 = list(dict5.keys())
    for k5 in key5:
        file2.write(f"LA_0000 {k5} - - {dict5[k5]}\n")

    key6 = list(dict6.keys())
    for k6 in key6:
        file2.write(f"LA_0000 {k6} - - {dict6[k6]}\n")

    key7 = list(dict7.keys())
    for k7 in key7:
        file2.write(f"LA_0000 {k7} - - {dict7[k7]}\n")

    key8 = list(dict8.keys())
    for k8 in key8:
        file2.write(f"LA_0000 {k8} - - {dict8[k8]}\n")









### Train set
import os

filename = "original_ASVspoof2019.LA.cm.train.trn.txt"
dict = {}

with open(filename, "r") as file1:
    for line in file1:
        line = line.strip().split()
        dict[line[1]] = line[4]
        
loc1 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/ambient_noise_old_incomplete/'
loc2 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/codec_losses/'
loc3 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/downsampling/'
loc4 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/gaussian_noise/'
loc5 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/packet_loss/'
loc6 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/reverberations/'
loc7 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/volume_reduction/'
loc8 = '/home/soumyas_kvmohan/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/'

loc1_files = os.listdir(loc1)
loc2_files = os.listdir(loc2)
loc3_files = os.listdir(loc3)
loc4_files = os.listdir(loc4)
loc5_files = os.listdir(loc5)
loc6_files = os.listdir(loc6)
loc7_files = os.listdir(loc7)
loc8_files = os.listdir(loc8)

dict1 = {}
for file in loc1_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict1[original_file] = dict[file]

dict2 = {}
for file in loc2_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict2[original_file] = dict[file]

dict3 = {}
for file in loc3_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict3[original_file] = dict[file]

dict4 = {}
for file in loc4_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict4[original_file] = dict[file]

dict5 = {}
for file in loc5_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict5[original_file] = dict[file]

dict6 = {}
for file in loc6_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict6[original_file] = dict[file]

dict7 = {}
for file in loc7_files:
    original_file = file[:-5]
    file = file[-17:-5]
    dict7[original_file] = dict[file]

dict8 = {}
for file in loc8_files:
    original_file = file[:-5]
    file = file[:-5]
    dict8[original_file] = dict[file]


with open("protocol.txt", "w") as file2:

    key1 = list(dict1.keys())
    for k1 in key1:
        file2.write(f"LA_0000 {k1} - - {dict1[k1]}\n")

    key2 = list(dict2.keys())
    for k2 in key2:
        file2.write(f"LA_0000 {k2} - - {dict2[k2]}\n")

    key3 = list(dict3.keys())
    for k3 in key3:
        file2.write(f"LA_0000 {k3} - - {dict3[k3]}\n")

    key4 = list(dict4.keys())
    for k4 in key4:
        file2.write(f"LA_0000 {k4} - - {dict4[k4]}\n")

    key5 = list(dict5.keys())
    for k5 in key5:
        file2.write(f"LA_0000 {k5} - - {dict5[k5]}\n")

    key6 = list(dict6.keys())
    for k6 in key6:
        file2.write(f"LA_0000 {k6} - - {dict6[k6]}\n")

    key7 = list(dict7.keys())
    for k7 in key7:
        file2.write(f"LA_0000 {k7} - - {dict7[k7]}\n")

    key8 = list(dict8.keys())
    for k8 in key8:
        file2.write(f"LA_0000 {k8} - - {dict8[k8]}\n")