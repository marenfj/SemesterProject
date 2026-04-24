import numpy as np
import soundfile
import torch
import torchaudio
import wavmark
from pathlib import Path

output_directory = Path("watermarked_audios/wavmark")
(output_directory / "bonafide").mkdir(parents=True, exist_ok=True)
(output_directory / "spoofed").mkdir(parents=True, exist_ok=True)

audiodir_bonafide = Path("datasets/ASVspoof5_partitions/audio_data/bonafide")
audiodir_spoofed = Path("datasets/ASVspoof5_partitions/audio_data/spoofed")

target_sample_rate = 16000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)
model.eval()

payload = np.random.choice([0, 1], size=16)
print("Payload:", payload)


def watermark_bonafide_audio() -> None:
    for audio_file_bonafide in audiodir_bonafide.iterdir():

        output_path = output_directory / "bonafide" / f"{audio_file_bonafide.stem}_watermarked.flac"
        if output_path.exists():
            continue

        wav, sr = torchaudio.load(str(audio_file_bonafide))
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            wav = resampler(wav)
            sr = target_sample_rate

        wav_bonafide = wav.mean(dim=0).numpy()
        with torch.no_grad():
            watermarked_bonafide, _ = wavmark.encode_watermark(model, wav_bonafide, payload, show_progress=False)
        watermarked_bonafide_tensor = torch.from_numpy(watermarked_bonafide).unsqueeze(0)

        torchaudio.save(str(output_path), watermarked_bonafide_tensor, sr)


def watermark_spoofed_audio() -> None:
    for audio_file_spoofed in audiodir_spoofed.iterdir():

        output_path = output_directory / "spoofed" / f"{audio_file_spoofed.stem}_watermarked.flac"
        if output_path.exists():
            continue

        wav, sr = torchaudio.load(str(audio_file_spoofed))
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            wav = resampler(wav)
            sr = target_sample_rate

        wav_spoofed = wav.mean(dim=0).numpy()
        with torch.no_grad():
            watermarked_spoofed, _ = wavmark.encode_watermark(model, wav_spoofed, payload, show_progress=False)
        watermarked_spoofed_tensor = torch.from_numpy(watermarked_spoofed).unsqueeze(0)

        torchaudio.save(str(output_path), watermarked_spoofed_tensor, sr)

def wavMark_watermark_audios() -> None:
    watermark_bonafide_audio()
    watermark_spoofed_audio()