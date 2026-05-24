from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal

from detection import (
    audioseal_score,
    load_audio_mono_resampled,
    ORIGINAL_ROOT,
    WATERMARKED_ROOT,
    AUDIOSEAL_DETECTOR_NAME,
)
from detection_metrics import roc_curve, auc_score, plot_roc
from device import get_device

OUTPUT_ROOT = Path("analysis_outputs")
RESIDUAL_CSV = OUTPUT_ROOT / "residual_detection_scores.csv"
PLAIN_WATERMARKS_DIR = Path("watermarked_audios/plain_watermarks/audioseal/bonafide")
WAVEFORM_PICTURE_DIR = Path("analysis_outputs/audioseal/bonafide/watermark_picture")
ROC_PATH = Path("analysis_outputs/audioseal/bonafide/ROC_isolated_watermark.png")
HIST_PATH = Path("analysis_outputs/audioseal/bonafide/score_distribution_isolated_watermark.png")
NUM_FILES = 100
DETECTION_THRESHOLD = 0.5


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal Error Rate: the point on the ROC where FPR == FNR (= 1 - TPR)."""
    fnr = 1.0 - tpr
    diffs = fpr - fnr
    idx = int(np.argmin(np.abs(diffs)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def plot_score_histogram(rows: list[dict], out_path: Path) -> None:
    orig = np.array([r["score_original"] for r in rows])
    res = np.array([r["score_residual"] for r in rows])
    wm = np.array([r["score_watermarked"] for r in rows])

    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(orig, bins=bins, alpha=0.6, label=f"clean original (mean={orig.mean():.3f})", color="tab:green")
    ax.hist(res, bins=bins, alpha=0.6, label=f"isolated watermark (mean={res.mean():.3f})", color="tab:orange")
    ax.hist(wm, bins=bins, alpha=0.6, label=f"watermarked audio (mean={wm.mean():.3f})", color="tab:blue")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="default threshold = 0.5")
    ax.set_xlabel("AudioSeal detection score")
    ax.set_ylabel("Number of files")
    ax.set_title(f"AudioSeal detection score distribution — {len(rows)} bonafide files")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_waveform_plot(residual: torch.Tensor, sr: int, stem: str, out_dir: Path) -> None:
    y = residual.squeeze(0).cpu().numpy()
    t = np.arange(y.shape[0]) / float(sr)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y, linewidth=0.5, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Isolated AudioSeal watermark — {stem}")
    ax.set_xlim(0, t[-1] if t.size else 0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_watermark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def iter_first_n_bonafide_audioseal_pairs(n: int):
    original_dir = ORIGINAL_ROOT / "bonafide"
    watermarked_dir = WATERMARKED_ROOT / "audioseal" / "bonafide"
    count = 0
    for original_file in sorted(original_dir.iterdir()):
        if not original_file.is_file():
            continue
        watermarked_file = watermarked_dir / f"{original_file.stem}_watermarked.flac"
        if not watermarked_file.exists():
            continue
        yield original_file, watermarked_file
        count += 1
        if count >= n:
            break


def run_residual_detection() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAIN_WATERMARKS_DIR.mkdir(parents=True, exist_ok=True)
    WAVEFORM_PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    detector = AudioSeal.load_detector(AUDIOSEAL_DETECTOR_NAME).to(get_device())
    detector.eval()

    rows: list[dict] = []
    pairs = list(iter_first_n_bonafide_audioseal_pairs(NUM_FILES))
    print(f"Processing {len(pairs)} bonafide pairs (AudioSeal residuals)...")

    for i, (original_file, watermarked_file) in enumerate(pairs, 1):
        wav_orig, sr = load_audio_mono_resampled(original_file)
        wav_wm, _ = load_audio_mono_resampled(watermarked_file)

        n_samples = min(wav_orig.shape[-1], wav_wm.shape[-1])
        wav_orig = wav_orig[..., :n_samples]
        wav_wm = wav_wm[..., :n_samples]

        residual = wav_wm - wav_orig

        residual_path = PLAIN_WATERMARKS_DIR / f"{original_file.stem}_residual.flac"
        torchaudio.save(str(residual_path), residual, sr)

        save_waveform_plot(residual, sr, original_file.stem, WAVEFORM_PICTURE_DIR)

        score_orig = audioseal_score(detector, wav_orig, sr)
        score_wm = audioseal_score(detector, wav_wm, sr)
        score_residual = audioseal_score(detector, residual, sr)

        peak = float(residual.abs().max())
        rms = float(torch.sqrt(torch.mean(residual ** 2)))

        rows.append({
            "stem": original_file.stem,
            "score_original": score_orig,
            "score_watermarked": score_wm,
            "score_residual": score_residual,
            "residual_peak": peak,
            "residual_rms": rms,
        })

        print(
            f"  [{i:3d}/{len(pairs)}] {original_file.stem}: "
            f"orig={score_orig:.4f}  wm={score_wm:.4f}  residual={score_residual:.4f}  "
            f"peak={peak:.4e}  rms={rms:.4e}"
        )

    fieldnames = [
        "stem",
        "score_original",
        "score_watermarked",
        "score_residual",
        "residual_peak",
        "residual_rms",
    ]
    with open(RESIDUAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    if n > 0:
        mean_orig = sum(r["score_original"] for r in rows) / n
        mean_wm = sum(r["score_watermarked"] for r in rows) / n
        mean_res = sum(r["score_residual"] for r in rows) / n
        print()
        print(f"Mean detection score on {n} files:")
        print(f"  original     = {mean_orig:.4f}")
        print(f"  watermarked  = {mean_wm:.4f}")
        print(f"  residual     = {mean_res:.4f}")

        tp = sum(1 for r in rows if r["score_residual"] >= DETECTION_THRESHOLD)
        fn = n - tp
        fp = sum(1 for r in rows if r["score_original"] >= DETECTION_THRESHOLD)
        tn = n - fp
        tpr = tp / n
        fpr = fp / n
        accuracy = (tp + tn) / (2 * n)
        print()
        print(f"Threshold-based detection @ {DETECTION_THRESHOLD}:")
        print(f"  TPR (residuals detected)        = {tp}/{n} = {tpr:.4f}")
        print(f"  FPR (clean flagged as wm)       = {fp}/{n} = {fpr:.4f}")
        print(f"  Accuracy                        = {tp + tn}/{2 * n} = {accuracy:.4f}")
        if fn > 0:
            missed = [r["stem"] for r in rows if r["score_residual"] < DETECTION_THRESHOLD]
            print(f"  Missed residuals ({fn}): {', '.join(missed)}")

    print(f"Wrote {len(rows)} rows to {RESIDUAL_CSV}")

    if rows:
        scores = np.array(
            [r["score_original"] for r in rows] + [r["score_residual"] for r in rows],
            dtype=float,
        )
        labels = np.array([0] * len(rows) + [1] * len(rows), dtype=int)
        fpr, tpr = roc_curve(labels, scores)
        auc = auc_score(labels, scores)
        eer = eer_from_roc(fpr, tpr)
        ROC_PATH.parent.mkdir(parents=True, exist_ok=True)
        plot_roc(
            {f"isolated watermark vs clean (EER={eer:.3f})": (fpr, tpr, auc)},
            "AudioSeal — detection of isolated watermark (residual) vs clean bonafide",
            ROC_PATH,
        )
        print(f"Wrote ROC plot to {ROC_PATH} (AUC={auc:.4f}, EER={eer:.4f})")

        plot_score_histogram(rows, HIST_PATH)
        print(f"Wrote histogram plot to {HIST_PATH}")

    return RESIDUAL_CSV


if __name__ == "__main__":
    run_residual_detection()
