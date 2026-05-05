from pathlib import Path
import csv
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
from device import get_device


TARGET_SR = 16000
AUDIOSEAL_DETECTOR_NAME = "audioseal_detector_16bits"

ORIGINAL_ROOT = Path("datasets/ASVspoof5_partitions/audio_data")
WATERMARKED_ROOT = Path("watermarked_audios")
OUTPUT_ROOT = Path("analysis_outputs")
WAVMARK_PAYLOAD_PATH = WATERMARKED_ROOT / "wavmark" / "payload.npy"
SCORES_CSV = OUTPUT_ROOT / "detection_scores.csv"


def load_audio_mono_resampled(path: Path, target_sr: int = TARGET_SR) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    return wav, sr


def audioseal_score(detector, wav: torch.Tensor, sr: int) -> float:
    """Mean fraction of frames where the detector flags watermark presence (in [0, 1])."""
    device = next(detector.parameters()).device
    with torch.no_grad():
        score, _ = detector.detect_watermark(wav.unsqueeze(0).to(device), sr)
    return float(score)


def wavmark_score(model, wav: torch.Tensor, payload: np.ndarray) -> float:
    """Bit accuracy of the decoded payload vs the reference payload (in [0, 1])."""
    audio_np = wav.mean(dim=0).numpy().astype(np.float32)
    with torch.no_grad():
        decoded, _ = wavmark.decode_watermark(model, audio_np, show_progress=False)
    if decoded is None:
        return 0.0
    decoded = np.asarray(decoded).astype(int)
    ref = np.asarray(payload).astype(int)
    n = min(len(decoded), len(ref))
    if n == 0:
        return 0.0
    return float(np.sum(decoded[:n] == ref[:n])) / float(n)


def _iter_pairs(model_dir_name: str):
    """Yield (dataset_type, stem, original_path, watermarked_path) for every paired file."""
    for dataset_type in ["bonafide", "spoofed"]:
        original_dir = ORIGINAL_ROOT / dataset_type
        watermarked_dir = WATERMARKED_ROOT / model_dir_name / dataset_type
        if not original_dir.exists() or not watermarked_dir.exists():
            continue
        for original_file in sorted(original_dir.iterdir()):
            if not original_file.is_file():
                continue
            stem = original_file.stem
            watermarked_file = watermarked_dir / f"{stem}_watermarked.flac"
            if not watermarked_file.exists():
                continue
            yield dataset_type, stem, original_file, watermarked_file


def collect_audioseal_scores() -> list[dict]:
    detector = AudioSeal.load_detector(AUDIOSEAL_DETECTOR_NAME).to(get_device())
    detector.eval()

    rows: list[dict] = []
    for dataset_type, stem, original_file, watermarked_file in _iter_pairs("audioseal"):
        wav_orig, sr = load_audio_mono_resampled(original_file)
        wav_wm, _ = load_audio_mono_resampled(watermarked_file)
        rows.append({
            "model": "audioseal",
            "dataset": dataset_type,
            "stem": stem,
            "label": 0,
            "score": audioseal_score(detector, wav_orig, sr),
        })
        rows.append({
            "model": "audioseal",
            "dataset": dataset_type,
            "stem": stem,
            "label": 1,
            "score": audioseal_score(detector, wav_wm, sr),
        })
    return rows


def collect_wavmark_scores() -> list[dict]:
    if not WAVMARK_PAYLOAD_PATH.exists():
        raise FileNotFoundError(
            f"Missing WavMark payload at {WAVMARK_PAYLOAD_PATH}. "
            "Run the wavmark watermarking stage first."
        )
    payload = np.load(WAVMARK_PAYLOAD_PATH)

    device = get_device()
    model = wavmark.load_model().to(device)
    model.eval()

    rows: list[dict] = []
    for dataset_type, stem, original_file, watermarked_file in _iter_pairs("wavmark"):
        wav_orig, _ = load_audio_mono_resampled(original_file)
        wav_wm, _ = load_audio_mono_resampled(watermarked_file)
        rows.append({
            "model": "wavmark",
            "dataset": dataset_type,
            "stem": stem,
            "label": 0,
            "score": wavmark_score(model, wav_orig, payload),
        })
        rows.append({
            "model": "wavmark",
            "dataset": dataset_type,
            "stem": stem,
            "label": 1,
            "score": wavmark_score(model, wav_wm, payload),
        })
    return rows


def run_detection() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print("Running AudioSeal detector...")
    rows.extend(collect_audioseal_scores())
    print("Running WavMark detector...")
    rows.extend(collect_wavmark_scores())

    fieldnames = ["model", "dataset", "stem", "label", "score"]
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {SCORES_CSV}")
    return SCORES_CSV


if __name__ == "__main__":
    run_detection()
