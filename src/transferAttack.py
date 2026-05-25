from pathlib import Path
import csv
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from audioseal import AudioSeal
from scipy.stats import gaussian_kde

from detection import (
    audioseal_score,
    load_audio_mono_resampled,
    ORIGINAL_ROOT,
    AUDIOSEAL_DETECTOR_NAME,
)
from detection_metrics import roc_curve, auc_score
from device import get_device


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)

NUM_SOURCE = 100   # residuals came from files x_1 .. x_100
NUM_TARGET = 100   # forge onto files x_101 .. x_200

PLAIN_WATERMARKS_DIR = Path("watermarked_audios/plain_watermarks/audioseal/bonafide")
OUTPUT_ROOT = Path("analysis_outputs")
FORGED_DIR = Path("watermarked_audios/transfer_attack/audioseal/bonafide")
ATTACK_CSV = OUTPUT_ROOT / "transfer_attack_scores.csv"
ATTACK_DENSITY_PATH = OUTPUT_ROOT / "transfer_attack_density.png"
DETECTION_THRESHOLD = 0.5


def iter_target_bonafide(skip: int, n: int):
    """Yield n bonafide files starting after the first `skip` sorted files."""
    original_dir = ORIGINAL_ROOT / "bonafide"
    all_files = sorted(f for f in original_dir.iterdir() if f.is_file())
    for f in all_files[skip : skip + n]:
        yield f


def load_residuals(n: int) -> list[tuple[str, torch.Tensor, int]]:
    """Load the first n residual .flac files from PLAIN_WATERMARKS_DIR, sorted."""
    residual_files = sorted(PLAIN_WATERMARKS_DIR.glob("*_residual.flac"))[:n]
    result = []
    for path in residual_files:
        wav, sr = load_audio_mono_resampled(path)
        stem = path.stem.replace("_residual", "")
        result.append((stem, wav, sr))
    return result


def plot_transfer_density(rows: list[dict], eer: float, out_path: Path) -> None:
    clean = np.array([r["score_target"] for r in rows], dtype=float)
    forged = np.array([r["score_forged"] for r in rows], dtype=float)

    x = np.linspace(0.0, 1.0, 500)
    kde_clean = gaussian_kde(clean, bw_method="scott")
    kde_forged = gaussian_kde(forged, bw_method="scott")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, kde_clean(x), color="red", linewidth=2, label="clean target bonafide")
    ax.plot(x, kde_forged(x), color="green", linewidth=2, label="forged (watermark transferred)")

    ax.set_xlabel("AudioSeal detection score", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(
        f"Transfer attack — AudioSeal detection on {len(rows)} forged bonafide files",
        fontsize=13,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_transfer_attack() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FORGED_DIR.mkdir(parents=True, exist_ok=True)

    detector = AudioSeal.load_detector(AUDIOSEAL_DETECTOR_NAME).to(get_device())
    detector.eval()

    residuals = load_residuals(NUM_SOURCE)
    targets = list(iter_target_bonafide(skip=NUM_SOURCE, n=NUM_TARGET))

    n_pairs = min(len(residuals), len(targets))
    print(f"Transfer attack: {n_pairs} residual-target pairs")

    rows: list[dict] = []
    for i, ((res_stem, residual, _res_sr), target_file) in enumerate(
        zip(residuals[:n_pairs], targets[:n_pairs]), 1
    ):
        wav_target, sr = load_audio_mono_resampled(target_file)

        # Trim both signals to the shorter length before adding
        n_samples = min(wav_target.shape[-1], residual.shape[-1])
        wav_target_trim = wav_target[..., :n_samples]
        residual_trim = residual[..., :n_samples]

        forged = wav_target_trim + residual_trim

        forged_path = FORGED_DIR / f"{target_file.stem}_forged.flac"
        torchaudio.save(str(forged_path), forged, sr)

        score_target = audioseal_score(detector, wav_target, sr)
        score_forged = audioseal_score(detector, forged, sr)

        rows.append({
            "target_stem": target_file.stem,
            "residual_stem": res_stem,
            "score_target": score_target,
            "score_forged": score_forged,
        })
        print(
            f"  [{i:3d}/{n_pairs}] {target_file.stem}: "
            f"clean={score_target:.4f}  forged={score_forged:.4f}"
        )

    fieldnames = ["target_stem", "residual_stem", "score_target", "score_forged"]
    with open(ATTACK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    if n > 0:
        mean_clean = sum(r["score_target"] for r in rows) / n
        mean_forged = sum(r["score_forged"] for r in rows) / n
        tp = sum(1 for r in rows if r["score_forged"] >= DETECTION_THRESHOLD)
        fp = sum(1 for r in rows if r["score_target"] >= DETECTION_THRESHOLD)

        scores = np.array(
            [r["score_target"] for r in rows] + [r["score_forged"] for r in rows], dtype=float
        )
        labels = np.array([0] * n + [1] * n, dtype=int)
        fpr, tpr = roc_curve(labels, scores)
        auc = auc_score(labels, scores)
        eer = eer_from_roc(fpr, tpr)

        print()
        print(f"Results over {n} pairs:")
        print(f"  Mean score clean target  : {mean_clean:.4f}")
        print(f"  Mean score forged        : {mean_forged:.4f}")
        print(f"  Attack success rate      : {tp}/{n} = {tp/n:.4f}  (forged detected as watermarked)")
        print(f"  False positive rate      : {fp}/{n} = {fp/n:.4f}  (clean flagged as watermarked)")
        print(f"  AUC                      : {auc:.4f}")
        print(f"  EER                      : {eer:.4f}")

        plot_transfer_density(rows, eer, ATTACK_DENSITY_PATH)
        print(f"Wrote density plot to {ATTACK_DENSITY_PATH}")

    print(f"Wrote {len(rows)} rows to {ATTACK_CSV}")
    return ATTACK_CSV


if __name__ == "__main__":
    run_transfer_attack()
