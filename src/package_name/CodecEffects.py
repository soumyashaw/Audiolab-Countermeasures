#!/usr/bin/python

import os
import torchaudio
from pydub import AudioSegment

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