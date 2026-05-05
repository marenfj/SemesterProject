from pathlib import Path
import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torchaudio


TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

ORIGINAL_ROOT = Path("datasets/ASVspoof5_partitions/audio_data")
WATERMARKED_ROOT = Path("watermarked_audios")
OUTPUT_ROOT = Path("analysis_outputs")

for model_type in ["audioseal", "wavmark"]:
    for dataset_type in ["bonafide", "spoofed"]:
        (OUTPUT_ROOT / model_type / dataset_type / "logmel").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / model_type / dataset_type / "stft").mkdir(parents=True, exist_ok=True)


def load_audio_mono_resampled(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    wav, sr = torchaudio.load(str(path))  # shape: (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    wav_np = wav.squeeze(0).numpy()
    return wav_np, sr


def compute_logmel_db(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_stft_db(y: np.ndarray) -> np.ndarray:
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )
    magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return stft_db


def compute_difference_map(original_spec: np.ndarray, watermarked_spec: np.ndarray) -> np.ndarray:
    min_frames = min(original_spec.shape[1], watermarked_spec.shape[1])
    return watermarked_spec[:, :min_frames] - original_spec[:, :min_frames]


def score_difference(diff_map: np.ndarray) -> float:
    return float(np.mean(np.abs(diff_map)))


def pair_files(model_type: str, dataset_type: str) -> list[tuple[Path, Path, str]]:
    """
    Returns list of:
    (original_path, watermarked_path, stem)
    """
    original_dir = ORIGINAL_ROOT / dataset_type
    watermarked_dir = WATERMARKED_ROOT / model_type / dataset_type

    pairs = []

    for original_file in original_dir.iterdir():
        if not original_file.is_file():
            continue

        watermarked_file = watermarked_dir / f"{original_file.stem}_watermarked.flac"
        if watermarked_file.exists():
            pairs.append((original_file, watermarked_file, original_file.stem))

    return pairs


def plot_triplet(
    original_spec: np.ndarray,
    watermarked_spec: np.ndarray,
    diff_spec: np.ndarray,
    sr: int,
    title_prefix: str,
    out_path: Path,
    representation: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if representation == "logmel":
        librosa.display.specshow(
            original_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=axes[0],
        )
        axes[0].set_title(f"{title_prefix} - Original")

        librosa.display.specshow(
            watermarked_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=axes[1],
        )
        axes[1].set_title(f"{title_prefix} - Watermarked")

        img = librosa.display.specshow(
            diff_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            cmap="coolwarm",
            ax=axes[2],
        )
        axes[2].set_title(f"{title_prefix} - Difference")
        fig.colorbar(img, ax=axes, format="%+2.0f dB")

    elif representation == "stft":
        librosa.display.specshow(
            original_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
            ax=axes[0],
        )
        axes[0].set_title(f"{title_prefix} - Original")

        librosa.display.specshow(
            watermarked_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
            ax=axes[1],
        )
        axes[1].set_title(f"{title_prefix} - Watermarked")

        img = librosa.display.specshow(
            diff_spec,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
            cmap="coolwarm",
            ax=axes[2],
        )
        axes[2].set_title(f"{title_prefix} - Difference")
        fig.colorbar(img, ax=axes, format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset_type(model_type: str, dataset_type: str) -> list[dict]:
    pairs = pair_files(model_type, dataset_type)
    results = []

    print(f"Found {len(pairs)} matched pairs for {model_type}/{dataset_type}")

    for i, (original_path, watermarked_path, stem) in enumerate(pairs, start=1):
        print(f"[{model_type}/{dataset_type}] Processing {i}/{len(pairs)}: {stem}")
        y_orig, sr_orig = load_audio_mono_resampled(original_path)
        y_wm, sr_wm = load_audio_mono_resampled(watermarked_path)
        assert sr_orig == sr_wm == TARGET_SR
        min_len = min(len(y_orig), len(y_wm))
        y_orig = y_orig[:min_len]
        y_wm = y_wm[:min_len]
        logmel_orig = compute_logmel_db(y_orig, TARGET_SR)
        logmel_wm = compute_logmel_db(y_wm, TARGET_SR)
        logmel_diff = compute_difference_map(logmel_orig, logmel_wm)
        logmel_score = score_difference(logmel_diff)
        stft_orig = compute_stft_db(y_orig)
        stft_wm = compute_stft_db(y_wm)
        stft_diff = compute_difference_map(stft_orig, stft_wm)
        stft_score = score_difference(stft_diff)

        results.append(
            {
                "model_type": model_type,
                "dataset_type": dataset_type,
                "stem": stem,
                "original_path": str(original_path),
                "watermarked_path": str(watermarked_path),
                "logmel_score": logmel_score,
                "stft_score": stft_score,
            }
        )

        plot_triplet(
            logmel_orig,
            logmel_wm,
            logmel_diff,
            TARGET_SR,
            title_prefix=f"{model_type}/{dataset_type}:{stem}",
            out_path=OUTPUT_ROOT / model_type / dataset_type / "logmel" / f"{stem}_logmel.png",
            representation="logmel",
        )

        plot_triplet(
            stft_orig,
            stft_wm,
            stft_diff,
            TARGET_SR,
            title_prefix=f"{model_type}/{dataset_type}:{stem}",
            out_path=OUTPUT_ROOT / model_type / dataset_type / "stft" / f"{stem}_stft.png",
            representation="stft",
        )

    return results


def save_rankings_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_top_examples(results: list[dict], model_type: str, dataset_type: str, score_key: str, top_k: int = 3) -> None:
    filtered = [r for r in results if r["model_type"] == model_type and r["dataset_type"] == dataset_type]
    filtered = sorted(filtered, key=lambda x: x[score_key], reverse=True)

    print(f"\nTop {top_k} {model_type}/{dataset_type} examples by {score_key}:")
    for row in filtered[:top_k]:
        print(f"  {row['stem']}: {row[score_key]:.4f}")


def spectral_analysis():
    all_results = []
    for model_type in ["audioseal", "wavmark"]:
        for dataset_type in ["bonafide", "spoofed"]:
            all_results.extend(analyze_dataset_type(model_type, dataset_type))

    save_rankings_csv(all_results, OUTPUT_ROOT / "rankings.csv")

    for model_type in ["audioseal", "wavmark"]:
        print_top_examples(all_results, model_type, "bonafide", "logmel_score", top_k=3)
        print_top_examples(all_results, model_type, "bonafide", "stft_score", top_k=3)
        print_top_examples(all_results, model_type, "spoofed", "logmel_score", top_k=3)
        print_top_examples(all_results, model_type, "spoofed", "stft_score", top_k=3)

