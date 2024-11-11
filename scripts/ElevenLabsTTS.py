########################### ElevenLabs TTS ##############################
import pandas as pd
from elevenlabs import save
from elevenlabs.client import ElevenLabs

# Open the file containing the target text and their file names
data = pd.read_csv("/Users/soumyashaw/validated.tsv", sep="\t")

ELEVENLABS_API_KEY = "sk_1fc7c1c9ac73a531cfae50e1e7143c493f1e0e572028869b"
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

response = client.voices.get_all()

globalCounter = 7428

"""for i in range(1):
    t = data['sentence'][globalCounter]
    if pd.isna(data['sentence'][globalCounter]):
        t = 'This is absurd!'
    print(t)
    audio = client.generate(text=t, voice=response.voices[19])

    save_file_path = f"/Users/soumyashaw/ElevenLabs/ElevenLabs_{globalCounter}.mp3"

    # Save the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    globalCounter += 1"""

for v in range(4, 20):
    for i in range(10):
        t = data['sentence'][globalCounter]
        if pd.isna(data['sentence'][globalCounter]):
            t = 'This is absurd!'
        print(t)
        audio = client.generate(text=t, voice=response.voices[v])

        save_file_path = f"/Users/soumyashaw/ElevenLabs/Dev/ElevenLabs_{globalCounter}.mp3"

        # Save the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        print(f"{save_file_path}: A new audio file was saved successfully!")

        globalCounter += 1