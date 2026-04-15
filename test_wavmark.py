import numpy as np
import torch
import torchaudio
import wavmark
from pathlib import Path

# Load a sample audio file
audio_file = Path("datasets/ASVspoof5_partitions/audio_data/bonafide/E_0000068801.flac")
wav, sr = torchaudio.load(str(audio_file))

print(f"Audio shape: {wav.shape}")
print(f"Audio dtype: {wav.dtype}")
print(f"Min value: {wav.min()}")
print(f"Max value: {wav.max()}")

# Resample if needed
target_sample_rate = 16000
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    wav = resampler(wav)

print(f"\nAfter resample - Min: {wav.min()}, Max: {wav.max()}")

payload = np.random.choice([0, 1], size=16)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

# Try with the original shape [1, 172865] (channels, samples)
print(f"\nTrying encode_watermark with shape {wav.shape}")
try:
    with torch.no_grad():
        watermarked = wavmark.encode_watermark(model, wav, payload, show_progress=False)
    print("Success! Shape:", watermarked.shape)
except Exception as e:
    print(f"Failed with error: {type(e).__name__}: {e}")

# Try with unsqueeze [1, 1, 172865] (batch, channels, samples)
print(f"\nTrying encode_watermark with shape {wav.unsqueeze(0).shape}")
try:
    with torch.no_grad():
        watermarked = wavmark.encode_watermark(model, wav.unsqueeze(0), payload, show_progress=False)
    print("Success! Shape:", watermarked.shape)
except Exception as e:
    print(f"Failed with error: {type(e).__name__}: {e}")
