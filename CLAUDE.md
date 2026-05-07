# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Spectrogram-based forensic analysis of audio watermarking. The pipeline takes audio from the ASVspoof5 dataset (bonafide + spoofed splits), watermarks each clip with two different models (AudioSeal and WavMark), then computes log-mel and STFT difference maps between original and watermarked audio to rank/visualize how perceptible each watermark is.

## Running the pipeline

```bash
# One-shot: rebuild venv, wipe outputs, install deps, run main
./startupscript.sh

# Manual run (must be invoked from the repo root — all paths are relative)
source .venv/bin/activate
python3.12 src/main.py
```

### Working directory matters

All input/output paths in `src/` are **relative** (`datasets/...`, `watermarked_audios/...`, `analysis_outputs/...`). Run scripts from the repo root, not from `src/`.

## Architecture

The pipeline is a three-stage chain orchestrated by `src/main.py`:

1. `audiosealWatermarking.watermark_audios()` — writes `watermarked_audios/audioseal/{bonafide,spoofed}/<stem>_watermarked.flac`
2. `wavmarkWatermarking.wavMark_watermark_audios()` — writes `watermarked_audios/wavmark/{bonafide,spoofed}/<stem>_watermarked.flac`
3. `spectral_analysis.spectral_analysis()` — pairs originals with watermarked outputs, computes log-mel and STFT spectrograms + difference maps, writes PNG triplets to `analysis_outputs/<model>/<dataset>/{logmel,stft}/` and a `rankings.csv` ranking files by mean abs difference.
4. detection.py - Detects wether the file was watermarked or not.
5. Detection_metrics.py - Writes the scores

### Module-level side effects (important)

Both watermarking modules execute heavy work at **import time**, not inside functions:

- `audiosealWatermarking.py` and `wavmarkWatermarking.py` each call `mkdir(parents=True, exist_ok=True)` for their output directories and load their respective ML model into memory the moment the module is imported.
- `wavmarkWatermarking.py` also generates and prints a fresh random 16-bit payload (`np.random.choice(...)`) on import. Every fresh process therefore embeds a different watermark — there is no seeding or persisted payload.

Consequence: simply importing `main` (or either watermarking module) loads AudioSeal/WavMark into memory and triggers a CUDA check. Be aware when refactoring or writing tests.

### Data layout assumptions

- Input root: `datasets/ASVspoof5_partitions/audio_data/{bonafide,spoofed}/*.flac`
- Audio is mono-resampled to 16 kHz before watermarking and before spectral analysis (`TARGET_SR = 16000`, `N_FFT = 1024`, `HOP_LENGTH = 256`, `N_MELS = 128`).
- Pairing in `spectral_analysis.pair_files` matches by filename stem and expects the suffix `_watermarked.flac`.
- The `wavmark` watermarker skips files whose output already exists (resumable); the `audioseal` watermarker overwrites unconditionally.

## Output directories are gitignored and ephemeral

`analysis_outputs/`, `watermarked_audios/`, and `datasets/` are in `.gitignore`. `startupscript.sh` deletes `analysis_outputs/` and `watermarked_audios/` on every run, so do not put anything you want to keep there.
