import os
import random
import pandas as pd
import soundfile as sf
from transformers import AutoProcessor, BarkModel

#os.environ["SUNO_USE_SMALL_MODELS"] = "True"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

data = pd.read_csv("/Users/soumyashaw/validated.tsv", sep="\t")

globalCounter = 7500

for i in range(250):
    text = data['sentence'][globalCounter].replace('"', '')

    voice_preset = f"v2/en_speaker_{random.randint(0, 9)}"

    inputs = processor(text, voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    sf.write(f"/Users/soumyashaw/Bark/Dev/bark_{globalCounter}.wav", audio_array, samplerate=24000)

    globalCounter += 1