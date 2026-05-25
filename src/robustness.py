from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

from detection import (
    audioseal_score,
    wavmark_score,
    load_audio_mono_resampled,
    TARGET_SR,
    ORIGINAL_ROOT,
    WATERMARKED_ROOT,
    AUDIOSEAL_DETECTOR_NAME,
    WAVMARK_PAYLOAD_PATH,
)
from audioseal import AudioSeal
import wavmark
from device import get_device

OUTPUT_ROOT = Path("analysis_outputs")
ROBUSTNESS_CSV = OUTPUT_ROOT / "robustness_scores.csv"

BANDS = [
    ("0-250Hz", 20, 250),
    ("250-500Hz", 250, 500),
    ("500-1kHz", 500, 1000),
    ("1k-2kHz", 1000, 2000),
    ("2k-4kHz", 2000, 4000),
    ("4k-8kHz", 4000, 7900),
]

def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def bandstop_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low_hz, high_hz], btype="bandstop", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def apply_filter(wav: torch.Tensor, sr: int, filter_fn, low_hz: float, high_hz: float) -> torch.Tensor:
    y = wav.squeeze(0).numpy()
    return torch.from_numpy(filter_fn(y, sr, low_hz, high_hz)).unsqueeze(0)

def _iter_pairs(model_dir_name: str, max_files: int = 15):
    for dataset_type in ["bonafide", "spoofed"]:
        original_dir = ORIGINAL_ROOT / dataset_type
        watermarked_dir = WATERMARKED_ROOT / model_dir_name / dataset_type
        if not original_dir.exists() or not watermarked_dir.exists():
            continue
        count = 0
        for original_file in sorted(original_dir.iterdir()):
            if not original_file.is_file():
                continue
            stem = original_file.stem
            watermarked_file = watermarked_dir / f"{stem}_watermarked.flac"
            if not watermarked_file.exists():
                continue
            yield dataset_type, stem, original_file, watermarked_file
            count += 1
            if count >= max_files:
                break

def _score(wav: torch.Tensor, sr: int, model_name: str, model, payload=None) -> float:
    if model_name == "audioseal":
        return audioseal_score(model, wav, sr)
    return wavmark_score(model, wav, payload)

def collect_robustness_scores(model_name: str, model, payload=None) -> list[dict]:
    rows = []
    pairs = list(_iter_pairs(model_name))
    for i, (dataset_type, stem, original_file, watermarked_file) in enumerate(pairs, 1):
        print(f"  [{model_name}/{dataset_type}] {i}/{len(pairs)}: {stem}")
        wav_orig, sr = load_audio_mono_resampled(original_file)
        wav_wm, _ = load_audio_mono_resampled(watermarked_file)

        for label, wav in [(0, wav_orig), (1, wav_wm)]:
            rows.append({
                "model": model_name, "dataset": dataset_type, "stem": stem,
                "label": label, "filter_type": "none", "band": "unfiltered",
                "score": _score(wav, sr, model_name, model, payload),
            })
            for band_name, low_hz, high_hz in BANDS:
                for filter_type, filter_fn in [("bandstop", bandstop_filter), ("bandpass", bandpass_filter)]:
                    wav_f = apply_filter(wav, sr, filter_fn, low_hz, high_hz)
                    rows.append({
                        "model": model_name, "dataset": dataset_type, "stem": stem,
                        "label": label, "filter_type": filter_type, "band": band_name,
                        "score": _score(wav_f, sr, model_name, model, payload),
                    })
    return rows

def plot_robustness(rows: list[dict], out_dir: Path) -> None:
    wm_rows = [r for r in rows if r["label"] == 1]
    band_labels = [b[0] for b in BANDS]

    for model in ["audioseal", "wavmark"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(f"{model} — robustness to frequency filtering (watermarked audio)")

        for ax, filter_type, title in [
            (axes[0], "bandstop", "Band-stop: band removed (attack)"),
            (axes[1], "bandpass", "Band-pass: only band kept (isolation)"),
        ]:
            for dataset in ["bonafide", "spoofed"]:
                baseline = np.mean([
                    r["score"] for r in wm_rows
                    if r["model"] == model and r["dataset"] == dataset and r["filter_type"] == "none"
                ] or [0.0])
                ax.axhline(baseline, linestyle="--", alpha=0.4, label=f"{dataset} baseline")

                means = [
                    np.mean([
                        r["score"] for r in wm_rows
                        if r["model"] == model and r["dataset"] == dataset
                        and r["filter_type"] == filter_type and r["band"] == band
                    ] or [0.0])
                    for band in band_labels
                ]
                ax.plot(band_labels, means, marker="o", label=dataset)
            
            ax.set_title(title)
            ax.set_xlabel("Frequency band")
            ax.set_ylabel("Mean detection score")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=30)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"robustness_{model}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved robustness_{model}.png")

def run_robustness() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("Loading AudioSeal detector...")
    detector = AudioSeal.load_detector(AUDIOSEAL_DETECTOR_NAME).to(get_device())
    detector.eval()
    print("Running AudioSeal robustness...")
    rows.extend(collect_robustness_scores("audioseal", detector))

    if not WAVMARK_PAYLOAD_PATH.exists():
        raise FileNotFoundError(f"Missing WavMark payload at {WAVMARK_PAYLOAD_PATH}.")
    payload = np.load(WAVMARK_PAYLOAD_PATH)
    print("Loading WavMark model...")
    wm_model = wavmark.load_model().to(get_device())
    wm_model.eval()
    print("Running WavMark robustness...")
    rows.extend(collect_robustness_scores("wavmark", wm_model, payload))

    fieldnames = ["model", "dataset", "stem", "label", "filter_type", "band", "score"]
    with open(ROBUSTNESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {ROBUSTNESS_CSV}")

    plot_robustness(rows, OUTPUT_ROOT)
    return ROBUSTNESS_CSV

if __name__ == "__main__":
    run_robustness()