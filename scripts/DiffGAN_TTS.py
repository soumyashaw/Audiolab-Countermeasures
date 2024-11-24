import os
import re
import pandas as pd

# Open the file containing the target text and their file names
data = pd.read_csv("/hkfs/home/haicore/hgf_cispa/hgf_yie2732/validated.tsv", sep="\t")

globalCounter = 3306

for i in range(695):
    text = data['sentence'][globalCounter].replace('"', '')

    command = f'python3 synthesize.py --text "{text}" --model shallow --restore_step 400000 --mode single --dataset LJSpeech'
    os.system(command)

    text = re.sub(r'[^A-Za-z0-9]+', '', text)
    
    if len(text) > 10:
        text = re.sub(r'[^A-Za-z0-9]+', '', text)[10]

    os.remove(f'/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TTS/DiffGAN-TTS/output/result/LJSpeech_shallow/400000/{text}.png')

    os.rename(f'/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TTS/DiffGAN-TTS/output/result/LJSpeech_shallow/400000/{text}.wav', f'/hkfs/home/haicore/hgf_cispa/hgf_yie2732/TTS/DiffGAN-TTS/output/result/LJSpeech_shallow/400000/diffgan_{globalCounter}.wav')

    globalCounter += 1