# SemesterProject — Spectrogram-based Forensic Analysis of Audio Watermarking

Semester project, spring 2026.

Investigates whether audio watermarks interact differently with **AI-generated (spoofed)** voices versus **genuine (bonafide)** voices.

## Pipeline

`src/main.py` orchestrates the core pipeline; `watermark_residual_detection.py` and `transferAttack.py` are run separately for the attack stages.

1. **AudioSeal embedding** (`audiosealWatermarking.py`) — writes `watermarked_audios/audioseal/{bonafide,spoofed}/<stem>_watermarked.flac`.
2. **WavMark embedding** (`wavmarkWatermarking.py`) — writes `watermarked_audios/wavmark/{bonafide,spoofed}/<stem>_watermarked.flac` plus a `payload.npy` reference.
3. **Spectral analysis** (`spectral_analysis.py`) — log-mel + linear STFT spectrograms and difference maps for each (original, watermarked) pair, plus a `rankings.csv` ordered by mean absolute difference.
4. **Detection** (`detection.py`) — runs each system's detector on originals and watermarked twins; writes per-clip scores to `analysis_outputs/detection_scores.csv`.
5. **Detection metrics** (`detection_metrics.py`) — accuracy, AUC, and ROC plots stratified by model and by dataset split.
6. **Watermark residual detection** (`watermark_residual_detection.py`) — isolates the raw AudioSeal watermark signal by computing `watermarked − original` for 100 bonafide files; saves residuals as `.flac` to `watermarked_audios/plain_watermarks/audioseal/bonafide/`; scores originals, watermarked audio, and residuals through the detector; writes `residual_detection_scores.csv`, a ROC curve, a score histogram, an EER density plot, and per-file waveform PNGs.
7. **Transfer attack** (`transferAttack.py`) — adds the 100 residuals from stage 6 onto 100 different (unwatermarked) bonafide target files; saves forged audio to `watermarked_audios/transfer_attack/audioseal/bonafide/`; scores clean targets vs. forged files and reports attack success rate, AUC, and EER; writes `transfer_attack_scores.csv` and a density plot.

## Running

```bash
# One-shot: rebuild venv, wipe outputs, install deps, run main
./startupscript.sh

# Manual run (must be invoked from the repo root — all paths are relative)
source .venv/bin/activate
TORCH_COMPILE_DISABLE=1 python3.12 src/main.py

```

## Layout

```

datasets/ASVspoof5_partitions/audio_data/{bonafide,spoofed}/ # input audio (gitignored)
watermarked_audios/{audioseal,wavmark}/{bonafide,spoofed}/ # embedding outputs (gitignored)
analysis_outputs/ # spectrograms, metrics, ROC plots (gitignored)
src/ # pipeline code

```

All audio is mono-resampled to 16 kHz before processing (`TARGET_SR = 16000`, `N_FFT = 1024`, `HOP_LENGTH = 256`, `N_MELS = 128`).

## Outputs

- `analysis_outputs/<model>/<dataset>/{logmel,stft}/` — PNG triplets (original | watermarked | difference) per clip.
- `analysis_outputs/rankings.csv` — clips ranked by mean absolute spectral difference.
- `analysis_outputs/detection_scores.csv` — per-clip detector scores with labels.
- `analysis_outputs/detection_metrics.csv` — accuracy + AUC per (model, dataset) group.
- `analysis_outputs/roc_*.png` — ROC curves (cross-model and cross-dataset).
- `analysis_outputs/residual_detection_scores.csv` — per-clip scores for original, watermarked, and isolated watermark (residual) audio (stage 6).
- `analysis_outputs/audioseal/bonafide/ROC_isolated_watermark.png` — ROC for residual vs. clean detection (stage 6).
- `analysis_outputs/audioseal/bonafide/score_distribution_isolated_watermark.png` — score histogram for original, watermarked, and residual audio (stage 6).
- `analysis_outputs/eer_density_plot.png` — KDE density plot of detection scores for clean vs. residual (stage 6).
- `analysis_outputs/audioseal/bonafide/watermark_picture/` — waveform PNGs of isolated watermark signals (stage 6).
- `watermarked_audios/plain_watermarks/audioseal/bonafide/` — isolated watermark residuals as `.flac` (stage 6, input to stage 7).
- `analysis_outputs/transfer_attack_scores.csv` — per-pair scores for clean target vs. forged (residual-injected) audio (stage 7).
- `analysis_outputs/transfer_attack_density.png` — KDE density plot comparing clean target vs. forged detection scores (stage 7).
- `watermarked_audios/transfer_attack/audioseal/bonafide/` — forged audio files with transferred watermark (stage 7).
