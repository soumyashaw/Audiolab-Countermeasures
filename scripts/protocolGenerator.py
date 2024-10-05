import os

bonafidePath = '/hkfs/home/haicore/hgf_cispa/hgf_yie2732/ASVspoof2019/LA/ASVspoof2019_LA_dev/dev_bonafide'
spoofPath = '/hkfs/home/haicore/hgf_cispa/hgf_yie2732/ASVspoof2019/LA/ASVspoof2019_LA_dev/dev_spoof'

bonafideAudio = os.listdir(bonafidePath)
spoofAudio = os.listdir(spoofPath)

with open("protocol.txt", "a") as f:
    for audio in bonafideAudio:
        f.write(f"LA_0000 {audio.split('.')[0]} - - bonafide\n")

    for audio in spoofAudio:
        f.write(f"LA_0000 {audio.split('.')[0]} - - spoof\n")


base_dir="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/ASVspoof2019/LA/ASVspoof2019_LA_train/wav48_silence_trimmed"

# Set the destination folder where all audio files will be moved
destination_folder="/hkfs/home/haicore/hgf_cispa/hgf_yie2732/ASVspoof2019/LA/ASVspoof2019_LA_train/wav48_silence_trimmed/all"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Find and move all .flac and .wav files to the destination folder
find "$base_dir" -type f \( -name "*.flac" -o -name "*.wav" \) -exec mv {} "$destination_folder" \;

echo "All audio files have been moved to $destination_folder"