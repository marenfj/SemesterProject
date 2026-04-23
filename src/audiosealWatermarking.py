from pathlib import Path
import torch
import torchaudio
from audioseal import AudioSeal

output_directory = Path("watermarked_audios/audioseal")
(output_directory / "bonafide").mkdir(parents=True, exist_ok=True)
(output_directory / "spoofed").mkdir(parents=True, exist_ok=True)
model_name = "audioseal_wm_16bits"
detector_name = "audioseal_detector_16bits"

target_sample_rate = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioSeal.load_generator(model_name)
model.eval()

audiodir_bonafide = Path("datasets/ASVspoof5_partitions/audio_data/bonafide")
audiodir_spoofed = Path("datasets/ASVspoof5_partitions/audio_data/spoofed")

bonified_original_vs_watermarked = {}
spoofed_original_vs_watermarked = {}


def watermark_bonafide_audio() -> None:
    for audio_file_bonafide in audiodir_bonafide.iterdir():

        wav, sr = torchaudio.load(str(audio_file_bonafide))
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            wav = resampler(wav)
            sr = target_sample_rate

        wav = wav.unsqueeze(0)
        with torch.no_grad():
            watermark_bonafide = model.get_watermark(wav)
            watermarked_bonafide_audio = wav + watermark_bonafide
        
        watermarked_bonafide_audio = watermarked_bonafide_audio.squeeze(0)

        output_path = output_directory / "bonafide" / f"{audio_file_bonafide.stem}_watermarked.flac"
        bonified_original_vs_watermarked[str(audio_file_bonafide)] = str(output_path)
        torchaudio.save(str(output_path), watermarked_bonafide_audio, sr)


def watermark_spoofed_audio() -> None:
    for audio_file_spoofed in audiodir_spoofed.iterdir():
        wav, sr = torchaudio.load(str(audio_file_spoofed))

        wav = wav.unsqueeze(0)
        with torch.no_grad():
            watermark_spoofed = model.get_watermark(wav)
            watermarked_spoofed_audio = wav + watermark_spoofed
        
        watermarked_spoofed_audio = watermarked_spoofed_audio.squeeze(0)

        output_path = output_directory / "spoofed" / f"{audio_file_spoofed.stem}_watermarked.flac"
        spoofed_original_vs_watermarked[str(audio_file_spoofed)] = str(output_path)
        torchaudio.save(str(output_path), watermarked_spoofed_audio, sr)

def watermark_audios() -> None:
    watermark_bonafide_audio()
    watermark_spoofed_audio()


 
