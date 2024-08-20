#!/usr/bin/python

import os

def add_reverberation(audioPath:str, targetpath: str, selectable: int = 0, iir_path: str = "/hkfs/home/haicore/hgf_cispa/hgf_yie2732/Audiolab-Countermeasures/data/iir.wav"):
    if selectable == 0:
        cmd1 = f"ffmpeg -loglevel error -i {audioPath} -map 0 -c:v copy -af aecho=1.0:0.9:70:0.5 {targetpath}"
        os.system(cmd1)
    elif selectable == 1:
        cmd2 = f"ffmpeg -loglevel error -i {audioPath} -i {iir_path} -filter_complex '[0] [1] afir=dry=10:wet=10 [reverb]; [0] [reverb] amix=inputs=2:weights=10 4' {targetpath}"
        os.system(cmd2)
