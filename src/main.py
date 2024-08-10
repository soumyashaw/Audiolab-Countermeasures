#!/usr/bin/python

import os
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
            print(len(target_files), len(reference_files))
            if target_files == reference_files:
                print("Equal")
            else:
                print("Not Equal")
            

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
    parser.add_argument('-t', '--target_dir', type=str, help="path to the target audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/BaselineDataset/LA/ASVspoof2019_LA_eval/lpf/")
    parser.add_argument('-r', '--reference_dir', type=str, help="path to the reference audio's directory", default="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/BaselineDataset/LA/ASVspoof2019_LA_eval/original_wav/")
    args = parser.parse_args()
    main()