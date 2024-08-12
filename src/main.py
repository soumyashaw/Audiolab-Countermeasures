#!/usr/bin/python

# Imports
import os
import argparse
from pesq import pesq
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import resample
from simple_term_menu import TerminalMenu

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
                
            print("Average STI: ", pesq_total/len(reference_files))
            

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
                print("Adding Gaussian Noise")
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
                break

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