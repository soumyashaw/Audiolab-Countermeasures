#!/usr/bin/python

import os
import shutil
from tqdm import tqdm
from utils import divide_list_randomly, make_directory, calculate_avg_sti

def add_reverberation(audioPath:str, targetpath: str, selectable: int = 0, iir_path: str = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/iir.wav"):
    if selectable == 0:
        cmd1 = f"ffmpeg -loglevel error -i {audioPath} -map 0 -c:v copy -af aecho=1.0:0.9:70:0.5 {targetpath}"
        os.system(cmd1)
    elif selectable == 1:
        cmd2 = f"ffmpeg -loglevel error -i {audioPath} -i {iir_path} -filter_complex '[0] [1] afir=dry=10:wet=10 [reverb]; [0] [reverb] amix=inputs=2:weights=10 4' {targetpath}"
        os.system(cmd2)

    return

def add_reverberation_effects(reference_dir: str, sti_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mAdding Reverberation\033[0m")
    print()

    # Collect the audio files from the reference directory
    audio_files = os.listdir(reference_dir)

    # Divide the audio files into two partitions for reverb effects
    audio_files = divide_list_randomly(audio_files, 2)

    # Change the directory to the reference directory
    os.chdir(reference_dir)

    os.chdir("../")

    # Make a directory to store the augmented data
    target_dir = os.getcwd() + "/augmented_data/reverberations/"
    make_directory(target_dir)

    for i in range(len(audio_files)):
        for audio in tqdm(audio_files[i], desc="Adding Reverberation to Partition " + str(i+1)):
            input_audio = reference_dir + str(audio)
            output_audio = os.getcwd() + "/augmented_data/reverberations/" + "reverb_" + str(audio)

            # Call the function to add reverb effects to the audio file
            add_reverberation(input_audio, output_audio, i)

            # Append the output audio file to the list for text file creation
            output_files.append("reverb_" + str(audio))

        avg_sti = calculate_avg_sti(audio_files[i], os.getcwd() + "/augmented_data/reverberations/", reference_dir, prefix = "reverb_")

        # Print the average STI for the packet drop rate
        print(f"Average STI for Reverberations Type {i+1}: {avg_sti}")

        if avg_sti < sti_threshold:
            print("\033[91mAverage STI is below the threshold.\033[0m Deleting augmented data.")

            # Remove the directory made
            shutil.rmtree(target_dir)

            break

    print()
    print("\033[92mReverberations added successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(reference_dir)
    os.chdir("../")

    with open('augmented_data/reverberations.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")

    return