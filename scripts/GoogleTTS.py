import pandas as pd
import random
from google.cloud import texttospeech

def text_to_speech(text, output_file):
    # Get a random voice
    random_voice = get_random_voice()

    # Setup synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure voice selection using the random voice
    voice = texttospeech.VoiceSelectionParams(
        language_code=random_voice.language_codes[0],  # Select the first language code available for the voice
        name=random_voice.name,
        ssml_gender=random_voice.ssml_gender
    )

    # Set the audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Instantiates a client and performs text-to-speech request
    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the speech to an MP3 file
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{output_file}" using voice: {random_voice.name}')

def list_voices():
    client = texttospeech.TextToSpeechClient()
    # Performs the list voices request
    response = client.list_voices()

    for voice in response.voices:
        print('Voice:')
        print(f'  Language codes: {voice.language_codes}')  # language_codes is a list
        print(f'  Name: {voice.name}')
        print(f'  SSML Gender: {voice.ssml_gender}')
        print(f'  Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}')

def get_random_voice():
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices()

    # Randomly select one voice from the list of available voices
    random_voice = random.choice(response.voices)
    return random_voice

# Open the file containing the target text and their file names
data = pd.read_csv("/Users/soumyashaw/validated.tsv", sep="\t")

globalCounter = 8250

for i in range(250):
    text = data['sentence'][globalCounter]

    text_to_speech(text, f'/Users/soumyashaw/GoogleTTS/Dev/googleTTS_{globalCounter}.mp3')

    globalCounter += 1