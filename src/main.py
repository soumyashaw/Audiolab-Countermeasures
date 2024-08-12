#!/usr/bin/python

import os
from scipy.io import wavfile
from pesq import pesq
import argparse
from simple_term_menu import TerminalMenu
from package_name.sti import stiFromAudio, readwav

def main():
    # Define the menu options
    menu_options = [
        "Calculate Average STI",
        "Trial 2",
        "Exit",
    ]

    # Create a TerminalMenu instance
    terminal_menu = TerminalMenu(menu_options, title="Main Menu", clear_screen=True)

    while True:
        # Show the menu and get the selected option
        selected_option_index = terminal_menu.show()

        # Perform actions based on the selected option
        if selected_option_index == 0:
            print("Calculating Average STI...")
            print("Target Directory: ", args.target_dir)
            print("Reference Directory: ", args.reference_dir)

            target_files = os.listdir(args.target_dir)
            reference_files = os.listdir(args.reference_dir)
            target_files.sort()
            reference_files.sort()

            counter = 0
            sti_total = 0.0

            for audio in reference_files:
                degrRate, target_Audio = wavfile.read(args.target_dir + str(audio))
                refRate, reference_audio = wavfile.read(args.reference_dir + str(audio))

                counter += 1
                PESQ = pesq(degrRate, reference_audio, target_Audio, 'wb')
                print(counter, " ", PESQ)
                
            print("Average STI: ", sti_total/len(reference_files))
            

        elif selected_option_index == 1:
            from datetime import datetime
            print(f"Current date and time: {datetime.now()}")

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