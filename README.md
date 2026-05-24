# ASR Robustness Under Acoustic Degradation

A controlled, reproducible evaluation harness for **Automatic Speech Recognition (ASR)
under realistic acoustic degradation** — additive noise, reverberation, and
telephone-grade codec distortion — plus research-grade analysis of how modern
ASR systems break down as conditions worsen.

## Motivation

Most public ASR benchmarks (LibriSpeech `test-clean`, etc.) use **clean, read
speech recorded with good microphones**. Real-world operational audio — the kind
the intelligence community actually has to transcribe — is **noisy, reverberant,
band-limited, and codec-compressed**. A model that scores 3% WER on read speech
can degrade catastrophically (and *silently*, via fluent hallucination) on
audio at 0 dB SNR.

This project asks: **how much, how, and why do ASR systems fail as audio
degrades — and does noise-aware fine-tuning recover the loss?**

## Research questions

1. **WER vs. SNR** — how steeply does each model's error rate climb as the
   signal-to-noise ratio drops? Where is the cliff?
2. **Degradation type** — additive noise vs. reverberation vs. narrowband
   telephone codec: which hurts most, and are the failure modes different?
3. **Hallucination under noise** — does Whisper emit confident, fluent, *wrong*
   transcripts on near-unintelligible audio? Quantified via insertion rate and
   output-length inflation.
4. **Robustness via fine-tuning** — does fine-tuning Whisper on
   noise-augmented data flatten the WER-vs-SNR curve? (multi-condition training
   ablation)
5. **Breakdowns** — does degradation interact with speaker accent and
   domain shift (read vs. conversational speech)?

## Approach

- **Degradation harness** (`src/asr_robustness/degrade/`): deterministic,
  parameterized degradation. Every degraded utterance records the *exact*
  parameters applied (SNR, noise clip, RIR, codec) so results are fully
  reproducible and sliceable.
- **Both synthetic and real degradation**: controlled synthetic degradation
  (clean corpus + MUSAN noise + measured RIRs at known SNRs) for clean ablation
  axes, validated against **VOiCES** — a corpus recorded in genuinely noisy,
  reverberant far-field conditions.
- **Multi-model benchmark**: Whisper family, wav2vec 2.0, and an ESPnet
  pretrained model, all scored through one harness.

## Repository layout

```
configs/            Experiment + degradation + model + training configs (YAML)
src/asr_robustness/
  audio.py          Audio I/O and resampling
  degrade/          The degradation harness (effects + pipeline)
  data/             Dataset download + manifest construction
  models/           Model registry + decoding adapters
  eval/             Decoding runner + WER scoring + breakdowns
  train/            Whisper fine-tuning (Phase 6 ablations)
  report/           Plots, tables, report generation
tests/              Unit tests
reports/            Generated research report
```

## Quick start

```bash
python3.11 -m venv speech_recognition      # already created
make setup                                 # install deps + package
make test                                  # run the unit suite
```

Reproduction pipeline (later phases): `make data` → `make decode` →
`make sweep` → `make report`.

## Status

Built incrementally in phases — see `make help` and the project tracker.

* Phases 1–4 (scaffold, degradation harness, data acquisition, model registry +
  WER scoring) — **done**.
* Phase 5 (degradation sweep): pilot done (`reports/pilot/`); full
  `snr_sweep.yaml` still pending.
* Phase 6 (fine-tuning ablations) — **code complete, awaiting cloud GPU runs**:
  * `src/asr_robustness/train/` — manifest-backed FT dataset with optional
    on-the-fly multi-condition augmentation, HF `Seq2SeqTrainer` driver.
  * `configs/train/clean_ft.yaml` — clean FT baseline on train-clean-100.
  * `configs/train/mct_ft.yaml` — multi-condition FT (same data, online noise
    + babble + reverb + codec + packet-loss augmentation across the SNR
    ladder).
  * `configs/experiments/ft_ablation.yaml` — re-evaluates the off-the-shelf,
    clean-FT, and MCT-FT checkpoints through the same condition grid as the
    Phase 5 pilot.

### Cloud GPU launch (Phase 6)

Training does not run on the Mac dev box (MPS lacks the kernels HF Trainer
needs for FP16/BF16 Whisper). The Phase 6 scripts are written to run on any
single CUDA GPU host (Lambda Labs, RunPod, Modal, Colab Pro). Launch recipe:

```bash
# 1. Provision a single-GPU box (A100/H100/L40 fits whisper-small with batch 16).
# 2. Clone the repo, install deps, build the train manifest:
make setup && make manifests
# 3. Stage the data: train-clean-100, dev-clean, MUSAN, RIRs.
make data
# 4. Run both ablations (each is ~4000 steps; ~1-2 h on A100):
make ft-clean
make ft-mct
# 5. Pull `results/checkpoints/whisper-small-{clean,mct}-ft/` back locally.
# 6. Run the head-to-head eval on the Mac (CPU/MPS is fine for inference):
make ft-eval
```
