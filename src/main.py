#!/usr/bin/python

# Imports
import os
import argparse
from tqdm import tqdm
from simple_term_menu import TerminalMenu
from package_name.sti import stiFromAudio, readwav

from package_name.VoIPEffects import add_voip_perterbation_effects
from package_name.TelephonyEffects import add_telephony_perterbation_effects
from package_name.VideoConferenceEffects import add_video_conferencing_perterbation_effects

from package_name.CodecEffects import add_codec_artifacts
from package_name.PacketLossEffects import add_packet_loss_effects
from package_name.DownsamplingEffects import add_downsampling_effects
from package_name.AmbientNoiseEffects import add_ambient_noise_effects
from package_name.ReverberationEffects import add_reverberation_effects
from package_name.GaussianNoiseEffects import add_gaussian_noise_effects
from package_name.AmplitudeReductionEffects import add_amplitude_reduction_effects

def main():
    # Define the menu options
    menu_options = [
        "Calculate Average STI",
        "Utility Based Data Augmentation",
        "Specific Perturbation Based Data Augmentation",
        "Exit",
    ]

    # Create a TerminalMenu instance
    terminal_menu = TerminalMenu(menu_options, title="Main Menu", clear_screen=True)

    while True:
        # Show the menu and get the selected option
        selected_option_index = terminal_menu.show()

        # Perform actions based on the selected option
        if selected_option_index == 0:
            print(" "*50 + "\033[91mCalculating Average STI\033[0m" + " "*50)
            print()
            print("Target Directory: ", args.target_dir)
            print("Reference Directory: ", args.reference_dir)

            target_files = os.listdir(args.target_dir)
            reference_files = os.listdir(args.reference_dir)
            target_files.sort()
            reference_files.sort()

            sti_total = 0.0

            for audio in tqdm(reference_files):
                target_Audio, degrRate  = readwav(args.target_dir + str(audio))
                reference_audio, refRate  = readwav(args.reference_dir + str(audio))

                try:
                    STI = stiFromAudio(reference_audio, target_Audio, refRate)
                    sti_total += STI

                except Exception as e:
                    print("Error in STI calculation:", e)
                    sti_total += 0.0
                    continue
                
            print("\033[91mAverage STI\033[0m: ", sti_total/len(reference_files))
            
        elif selected_option_index == 1:
            utility_data_augmentation_menu_options = [
                "Telephony Perturbation Effects",
                "VoIP Perturbation Effects",
                "Video Conferencing Perturbation Effects",
                "Go Back"
            ]

            utility_data_augmentation_menu = TerminalMenu(utility_data_augmentation_menu_options, title="Utility Based Data Augmentation Menu", clear_screen=True)
            utility_data_augmentation_selected_option_index = utility_data_augmentation_menu.show()

            if utility_data_augmentation_selected_option_index == 0:
                add_telephony_perterbation_effects()

            if utility_data_augmentation_selected_option_index == 1:
                gaussian_SNR_levels = [5, 7, 9, 11, 13, 15]
                ambient_SNR_levels = [5, 10, 15, 20, 25, 30]
                packet_loss_rates = [0.02, 0.05, 0.07, 0.01]
                add_voip_perterbation_effects(gaussian_SNR_levels, ambient_SNR_levels, args.ambient_noise_dir, args.volume_threshold, args.lower_sampling_rate, args.current_sampling_rate, packet_loss_rates, args.reference_dir, args.sti_threshold)

            if utility_data_augmentation_selected_option_index == 2:
                add_video_conferencing_perterbation_effects()

            if utility_data_augmentation_selected_option_index == 3:
                continue


        elif selected_option_index == 2:

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
                #SNR_levels_dB = [5, 10, 15, 20, 25, 30]
                SNR_levels_dB = [5, 6, 7, 8, 9, 10]
                add_gaussian_noise_effects(SNR_levels_dB, args.reference_dir, args.sti_threshold)

            elif augment_data_selected_option_index == 1:
                SNR_levels_dB = [5, 10, 15, 20, 25, 30]
                add_ambient_noise_effects(SNR_levels_dB, args.reference_dir, args.ambient_noise_dir, args.sti_threshold)
                
            elif augment_data_selected_option_index == 2:
                add_reverberation_effects(args.reference_dir, args.sti_threshold)

            elif augment_data_selected_option_index == 3:
                add_amplitude_reduction_effects(args.reference_dir, args.volume_threshold)
                
            elif augment_data_selected_option_index == 4:
                add_codec_artifacts(args.reference_dir)

            elif augment_data_selected_option_index == 5:
                add_downsampling_effects(args.reference_dir, args.lower_sampling_rate, args.current_sampling_rate, args.sti_threshold)

            elif augment_data_selected_option_index == 6:
                add_packet_loss_effects(args.reference_dir, args.packet_loss_rate, args.sti_threshold)

            elif augment_data_selected_option_index == 6:
                continue

        elif selected_option_index == 3:
            print("Terminating Execution. Goodbye!")
            break

        # Pause to allow the user to read the output
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--target_dir', type=str, help="path to the target audio's directory", default="/home/soumyas_kvmohan/ASVspoof_train/train_spoof/ambient_noise/")
    parser.add_argument('-r', '--reference_dir', type=str, help="path to the reference audio's directory", default="/home/soumyas_kvmohan/ASVspoof_train/train_spoof/ambient_noise/")
    parser.add_argument('-s', '--sti_threshold', type=float, help="STI threshold for the augmented data", default=0.5)
    parser.add_argument('-v', '--volume_threshold', type=float, help="Volume threshold for the augmented data", default=-35)
    parser.add_argument('-l', '--packet_loss_rate', type=float, help="Target Packet Loss Rate for the augmented data", default=0.1)
    parser.add_argument('-m', '--lower_sampling_rate', type=int, help="Lower bound sampling rate to be applied to the audios", default=3400)
    parser.add_argument('-e', '--current_sampling_rate', type=int, help="Current sampling rate of the audio files", default=16000)
    parser.add_argument('-i', '--input_format', type=str, help="Input format of the audio files", default="wav")
    parser.add_argument('-o', '--output_format', type=str, help="Output format of the audio files", default="wav")
    parser.add_argument('-n', '--ambient_noise_dir', type=str, help="path to the ambient noise files to be used", default="/home/soumyas_kvmohan/Audiolab-Countermeasures/data/ambient_noise/")
    args = parser.parse_args()
    main()