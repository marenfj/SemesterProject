# Spectrogram-based Forensic Analysis of Audio Watermarking

EURECOM semester project, spring 2026.

This project investigates a single question: **do audio watermarks behave differently when
embedded into AI-generated (spoofed) speech versus genuine (bonafide) speech?** If a
watermark leaves a distinct, more (or less) detectable trace on synthetic voices, that has
direct implications for using watermarking as a deepfake-provenance tool.

To explore this, the project takes clips from the **ASVspoof5** dataset, embeds two
different audio watermarking systems into each clip, and then measures the watermark two
ways:

1. **Spectrally** — how visible is the watermark in log-mel and STFT spectrograms,
   measured as the difference between the original and watermarked audio?
2. **By detection** — how reliably does each watermark's own detector recover the
   watermark, and does that reliability differ between bonafide and spoofed audio?

Everything is stratified along two axes throughout: the **watermarking model**
(`audioseal` / `wavmark`) and the **dataset split** (`bonafide` / `spoofed`).

## The two watermarking systems

| System                                                         | How the watermark is embedded                                                            | What "detection score" means                                                         |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **[AudioSeal](https://github.com/facebookresearch/audioseal)** | A generator network produces an additive watermark signal that is added to the waveform. | Watermark-**presence probability** in `[0, 1]` from AudioSeal's detector.            |
| **[WavMark](https://github.com/wavmark/wavmark)**              | A 16-bit payload is encoded into the waveform by the WavMark model.                      | **Bit accuracy** in `[0, 1]` of the decoded payload vs. the known reference payload. |

## Pipeline

`src/main.py` runs a five-stage pipeline. Each stage is one module under `src/`, and stages
communicate **through files on disk** rather than in memory — so any stage can be re-run on
its own without redoing the others.

```
                          ASVspoof5 input clips
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
   1. audiosealWatermarking.py              2. wavmarkWatermarking.py
              │                                          │
              └────────────────────┬────────────────────┘
                                   ▼
                       3. spectral_analysis.py
                                   │
                                   ▼
                          4. detection.py
                                   │
                                   ▼
                       5. detection_metrics.py
```

### Stage details

1. **AudioSeal embedding** — `audiosealWatermarking.py`
   Loads the AudioSeal generator, embeds a watermark into every `bonafide` and `spoofed`
   clip, and writes `watermarked_audios/audioseal/<split>/<stem>_watermarked.flac`.
   Re-encodes on every run.

2. **WavMark embedding** — `wavmarkWatermarking.py`
   Generates a random 16-bit payload **once** and persists it to
   `watermarked_audios/wavmark/payload.npy` (reused on later runs); the detection stage
   compares against this exact payload. Writes `..._watermarked.flac` per clip and
   **skips files that already exist**, so this stage is resumable.

3. **Spectral analysis** — `spectral_analysis.py`
   For each `(original, watermarked)` pair, computes log-mel and linear-STFT spectrograms
   (in dB) plus a difference map, saves a three-panel PNG (original | watermarked |
   difference), and records the mean absolute difference. Writes
   `analysis_outputs/rankings.csv` ranking every clip by how much the watermark perturbed
   the spectrum.

4. **Detection** — `detection.py`
   Runs each system's detector on the originals (`label = 0`) and their watermarked twins
   (`label = 1`), writing one row per clip per label to
   `analysis_outputs/detection_scores.csv` with columns
   `model, dataset, stem, label, score`.

5. **Detection metrics** — `detection_metrics.py`
   Aggregates the scores into accuracy and AUC per `(model, dataset)` group, writes
   `analysis_outputs/detection_metrics.csv`, and renders ROC plots comparing
   AudioSeal vs. WavMark within each split and bonafide vs. spoofed within each model.

Three further experiments run independently of the main pipeline:

6. **Robustness sweep** — `robustness.py`
   Applies band-pass / band-stop filters across a set of frequency bands to the watermarked
   audio and re-scores it, revealing which bands the watermark survives in. Writes
   `analysis_outputs/robustness_scores.csv` and `robustness_<model>.png` plots.

7. **Residual detection** — `watermark_residual_detection.py`
   An AudioSeal-only experiment on bonafide files. Computes the pure watermark signal
   (`watermarked − original`) for 100 files, saves each residual as a `.flac` file under
   `watermarked_audios/plain_watermarks/audioseal/bonafide/`, and scores the original,
   watermarked, and isolated residual through the AudioSeal detector. Reports threshold-based
   TPR/FPR/accuracy, AUC, and EER, and writes a score histogram, per-file waveform plots, a
   KDE density plot, and an ROC plot — together showing how detectable the watermark signal
   is when completely stripped from the host audio.

8. **Transfer attack** — `transferAttack.py`
   An AudioSeal-only experiment that tests whether a watermark residual extracted from one
   clip can be transplanted onto a different, unwatermarked clip to fool the detector.
   Takes the 100 residuals produced by Stage 7, adds each one to a fresh, unwatermarked
   bonafide clip (files 101–200 in the sorted bonafide set), and scores both the clean target
   and the forged audio with the AudioSeal detector. Reports attack success rate,
   false-positive rate, AUC, and EER, and writes a KDE density plot comparing the score
   distributions of clean versus forged clips. **Requires Stage 7 to have been run first**
   (the residual `.flac` files must exist on disk).

## Repository layout

```
src/                      pipeline code (one module per stage)
datasets/                 ASVspoof5 input clips and protocol files
watermarked_audios/       watermarked .flac output, one folder per model
analysis_outputs/         spectrograms, CSVs, and ROC plots
requirements.txt          Python dependencies
startupscript.sh          one-shot setup + run
```

## Requirements

- **Python 3.12**
- The packages in `requirements.txt` (installed by the steps below).

The first run will download the AudioSeal and WavMark model weights, so it needs network
access.

## Running

### Option A — one shot

```bash
./startupscript.sh
```

This creates the `.venv`, **deletes any existing `analysis_outputs/` and
`watermarked_audios/`**, installs `requirements.txt`, and runs the full pipeline. Use this
for a clean rebuild; avoid it if you want to keep previous outputs.

### Option B — manual

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
TORCH_COMPILE_DISABLE=1 python3.12 src/main.py
```

Two rules make the manual run work:

- **Run from the repository root.** Every input/output path in the code is relative to the
  current directory.
- **Run modules as `src/<file>.py`** (not from inside `src/`). The modules import each other
  by bare name, which only resolves when `src/` is the script directory.
- **Keep `TORCH_COMPILE_DISABLE=1`.** Without it the watermarking models try to use
  `torch.compile` and fail.

You can run any stage on its own (each module has a `__main__` entry point), as long as its
inputs already exist on disk:

```bash
TORCH_COMPILE_DISABLE=1 python3.12 src/spectral_analysis.py
TORCH_COMPILE_DISABLE=1 python3.12 src/detection.py
TORCH_COMPILE_DISABLE=1 python3.12 src/detection_metrics.py
TORCH_COMPILE_DISABLE=1 python3.12 src/robustness.py
TORCH_COMPILE_DISABLE=1 python3.12 src/watermark_residual_detection.py

# Transfer attack (requires watermark_residual_detection.py to have been run first)
TORCH_COMPILE_DISABLE=1 python3.12 src/transferAttack.py
```

Hardware is selected automatically (`src/device.py`): CUDA if available, then Apple Silicon
MPS, then CPU.

## Outputs

Everything lands under `analysis_outputs/`:

| Path                                         | Contents                                                       |
| -------------------------------------------- | -------------------------------------------------------------- |
| `<model>/<dataset>/logmel/<stem>_logmel.png` | log-mel triplet: original \| watermarked \| difference         |
| `<model>/<dataset>/stft/<stem>_stft.png`     | STFT triplet (same layout)                                     |
| `rankings.csv`                               | every clip ranked by mean absolute spectral difference         |
| `detection_scores.csv`                       | per-clip detector scores: `model, dataset, stem, label, score` |
| `detection_metrics.csv`                      | accuracy + AUC per `(model, dataset)` group                    |
| `roc_models_<dataset>.png`                   | ROC: AudioSeal vs. WavMark within one split                    |
| `roc_datasets_<model>.png`                   | ROC: bonafide vs. spoofed within one model                     |

The robustness, residual, and transfer-attack experiments add the following under
`analysis_outputs/`:

| Path                                                                      | Contents                                                                            |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `robustness_scores.csv`                                                   | per-band detection scores after filtering                                           |
| `robustness_<model>.png`                                                  | robustness plot per watermarking model                                              |
| `residual_detection_scores.csv`                                           | per-file scores: original / watermarked / isolated residual + peak and RMS          |
| `audioseal/bonafide/ROC_isolated_watermark.png`                           | ROC for detecting the isolated watermark vs. clean bonafide audio                   |
| `audioseal/bonafide/score_distribution_isolated_watermark.png`            | histogram: original / residual / watermarked audio detection scores                 |
| `eer_density_plot.png`                                                    | KDE density: clean bonafide vs. isolated watermark detection scores                 |
| `audioseal/bonafide/watermark_picture/<stem>_watermark.png`               | waveform of the isolated AudioSeal watermark per file                               |
| `transfer_attack_scores.csv`                                              | per-pair scores: `target_stem, residual_stem, score_target, score_forged`           |
| `transfer_attack_density.png`                                             | KDE density: clean target vs. forged (watermark transferred) detection scores       |

## Audio processing notes

All audio is treated uniformly so spectrograms and detectors are comparable: converted to
**mono**, resampled to **16 kHz**, and analyzed with `N_FFT = 1024`, `HOP_LENGTH = 256`,
`N_MELS = 128`. Original and watermarked clips are length-matched (truncated to the shorter
of the two) before any comparison. Pairing is by filename **stem**: `<stem>.flac` is matched
to `<stem>_watermarked.flac`, and unmatched files are skipped.

## Who did what?

**Maren** built the watermarking stages that embed AudioSeal and WavMark into the audio,
the robustness attack that filters the watermarked clips and analyses how well the watermark
survives across frequency bands, and the transfer attack that extracts watermark residuals
and transplants them onto unwatermarked clips to test whether the detector can be fooled.
**Magnus** built the spectral analysis comparing originals to their watermarked twins, and
the detection and metrics stages that score and evaluate each system. Separating the
watermark from the audio (the residual experiment) was done by **both** of us.
