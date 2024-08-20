#!/usr/bin/python

import os
import random
import shutil
from tqdm import tqdm
from package_name.sti import stiFromAudio, readwav

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

def calculate_avg_sti(target_audio_list, target_dir, reference_dir, prefix = ""):

    sti_total = 0.0

    for audio in tqdm(target_audio_list, desc="Calculating Average STI"):
        target_Audio, degrRate  = readwav(target_dir + prefix + str(audio))
        reference_audio, refRate  = readwav(reference_dir + str(audio))

        try:
            STI = stiFromAudio(reference_audio, target_Audio, refRate)
            sti_total += STI

        except Exception as e:
            print("Error in STI calculation:", e)
            sti_total += 0.0
            continue
        
    return sti_total/len(target_audio_list)