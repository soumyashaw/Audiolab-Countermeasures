import os

bonafidePath = '/home/soumyas_kvmohan/sampledASVspoof2021/LA/ASVspoof2019_LA_eval/real'
spoofPath = '/home/soumyas_kvmohan/sampledASVspoof2021/LA/ASVspoof2019_LA_eval/fake'

bonafideAudio = os.listdir(bonafidePath)
spoofAudio = os.listdir(spoofPath)

with open("protocol.txt", "a") as f:
    for audio in bonafideAudio:
        f.write(f"LA_0000 {audio.split('.')[0]} - - bonafide\n")

    for audio in spoofAudio:
        f.write(f"LA_0000 {audio.split('.')[0]} - - spoof\n")


base_dir="/home/soumyas_kvmohan/LibriSpeech/LibriSpeech/train-clean-100"

# Set the destination folder where all audio files will be moved
destination_folder="/home/soumyas_kvmohan/LibriSpeech/LibriSpeech/train-clean-100"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Find and move all .flac and .wav files to the destination folder
find "$base_dir" -type f \( -name "*.flac" -o -name "*.wav" \) -exec mv {} "$destination_folder" \;

echo "All audio files have been moved to $destination_folder