#!/usr/bin/python

import os
import shutil
import torchaudio
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
from package_name.utils import divide_list_randomly, make_directory

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
        return encoder.apply(waveform, sr)
    
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
    
def add_codec_artifacts(reference_dir: str):
    output_files = []

    print(" "*50 + "\033[91mAdding Codec Losses\033[0m")
    print()

    codecs = ['mulaw', 'g722', 'alaw', 'opus']

    # Empty list to store the Gaussian Noise directories with particular SNR levels
    directories_made = []
    
    # Change the directory to the reference directory
    os.chdir(reference_dir)

    # List all the audio files in the reference directory (original audio files)
    reference_files = os.listdir(reference_dir)

    # Divide the list of audio files into n partitions (based on the number of SNR levels)
    audio_files = divide_list_randomly(reference_files, len(codecs))

    # Check the existence of directory to store the augmented data exists
    for i in range(len(codecs)):
        # Change the directory to the reference directory
        os.chdir(reference_dir)

        # Create a new directory to store the augmented data
        os.chdir("../")
        target_dir = os.getcwd() + "/augmented_data/" + str(codecs[i]) + "/"
        make_directory(target_dir)
        directories_made.append(target_dir)

        # Add Gaussian Noise to the audio files with given SNR level
        for audio in tqdm(audio_files[i], desc="Adding Codec Losses to Partition " + str(i+1)):
            input_audio = reference_dir + str(audio)

            # Append the identifier string to output audio file
            output_audio = target_dir + str(codecs[i]) + "_" + str(audio)

            # Append the output audio file to the list for text file creation
            output_files.append(str(codecs[i]) + "_" + str(audio))

            # Call the function to add Codec losses to the audio file
            codec_added_audio = add_codec_loss(input_audio, "wav", codecs[i])

            if codecs[i] == 'opus':
                codec_added_audio.export(output_audio, format="wav")

            else:
                sf.write(output_audio, codec_added_audio, 16000)

    print()
    print("\033[92mCodec Artifacts added successfully!\033[0m")

    # Create a text file to store the output audio files
    os.chdir(reference_dir)
    os.chdir("../")

    with open('augmented_data/codec_losses.txt', 'w') as file:
        for item in output_files:
            file.write(f"{item}\n")

    # Cleanup: Merge the directories into one
    current_path = os.getcwd() + "/augmented_data/"
    make_directory(current_path + "codec_losses/")
    for path in directories_made:
        for file in os.listdir(path):
            shutil.move(path + file, current_path + "codec_losses/" + file)
        os.rmdir(path)