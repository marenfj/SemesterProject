# SemesterProject — Spectrogram-based Forensic Analysis of Audio Watermarking

Semester project, spring 2026.

Investigates whether audio watermarks interact differently with **AI-generated (spoofed)** voices versus **genuine (bonafide)** voices.

## Pipeline

`src/main.py` orchestrates a five-stage chain:

1. **AudioSeal embedding** (`audiosealWatermarking.py`) — writes `watermarked_audios/audioseal/{bonafide,spoofed}/<stem>_watermarked.flac`.
2. **WavMark embedding** (`wavmarkWatermarking.py`) — writes `watermarked_audios/wavmark/{bonafide,spoofed}/<stem>_watermarked.flac` plus a `payload.npy` reference.
3. **Spectral analysis** (`spectral_analysis.py`) — log-mel + linear STFT spectrograms and difference maps for each (original, watermarked) pair, plus a `rankings.csv` ordered by mean absolute difference.
4. **Detection** (`detection.py`) — runs each system's detector on originals and watermarked twins; writes per-clip scores to `analysis_outputs/detection_scores.csv`.
5. **Detection metrics** (`detection_metrics.py`) — accuracy, AUC, and ROC plots stratified by model and by dataset split.

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
