#!/usr/bin/python

import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from utils import make_directory, calculate_avg_sti

def simulate_packet_loss(audio_data, loss_rate):
    num_samples = len(audio_data)
    lost_samples = int(loss_rate * num_samples)
    indices_to_drop = np.random.choice(num_samples, lost_samples, replace=False)

    simulated_data = np.delete(audio_data, indices_to_drop)
    return simulated_data

def add_packet_loss_effects(reference_dir: str, packet_loss_rate: float, sti_threshold: float):
    output_files = []

    print(" "*50 + "\033[91mPacket Loss Effects\033[0m")
    print()

    audio_files = os.listdir(reference_dir)

    os.chdir(reference_dir)
    os.chdir("../")

    print(os.getcwd())

    target_dir = os.getcwd() + "/augmented_data/packet_loss/"
    make_directory(target_dir)

    for audio in tqdm(audio_files, desc="Adding Packet Loss Effects"):
        input_audio = reference_dir + str(audio)
        output_audio = target_dir + "pl_" + str(audio)

        # Append the output audio file to the list for text file creation
        output_files.append("pl_" + str(audio))

        reference_audio, sr = librosa.load(input_audio, sr=None)

        loss_rate = packet_loss_rate
        packet_loss_audio = simulate_packet_loss(reference_audio, loss_rate)

        sf.write(output_audio, packet_loss_audio, sr)

    avg_sti = calculate_avg_sti(audio_files, target_dir, reference_dir, prefix = "pl_")

    # Print the average STI for the packet drop rate
    print(f"Average STI for {loss_rate} packet drop rate: {avg_sti}")

    if avg_sti < sti_threshold:
        print("\033[91mAverage STI is below the threshold.\033[0m Deleting augmented data.")

        # Remove the directory made
        shutil.rmtree(target_dir)

    print()
    print("\033[92mPacket Loss Effect added successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(reference_dir)
    os.chdir("../")

    with open('augmented_data/packet_loss.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")
