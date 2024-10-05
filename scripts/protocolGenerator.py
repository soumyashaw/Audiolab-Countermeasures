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