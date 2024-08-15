import librosa
import soundfile as sf
from torchaudio.utils import download_asset

SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
noise_audio, sr = librosa.load(SAMPLE_NOISE, sr=None)

sf.write("noise1.wav", noise_audio, sr)