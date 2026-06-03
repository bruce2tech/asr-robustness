# ASR Robustness Under Acoustic Degradation

A controlled, reproducible evaluation harness for **Automatic Speech Recognition (ASR)
under realistic acoustic degradation** — additive noise, multi-talker babble,
reverberation, telephony / VoIP codecs, and packet loss — plus a multi-condition
fine-tuning ablation that targets the hallucination failure mode.

## Findings (v1.0)

**Full writeup:** [reports/REPORT.md](reports/REPORT.md) · **Auto-generated tables:**
[reports/pilot/summary.md](reports/pilot/summary.md) ·
[reports/ft_ablation/summary.md](reports/ft_ablation/summary.md)

Three results from a controlled head-to-head on 100 `test-clean` utterances × 16
realistic conditions:

1. **Babble is a categorically different kind of noise.** At 0 dB SNR, switching
   from stationary noise to multi-talker babble multiplies Whisper-base WER from
   29.0 % to 175.0 % — a ~6× increase at the same physical signal-to-noise ratio.
2. **Whisper and wav2vec 2.0 fail in opposite directions.** Under heavy degradation
   Whisper over-generates fluent hallucinated text (length up to 22× the
   reference); wav2vec 2.0 under-generates (drops words). Comparable WER,
   fundamentally different operational risk — *plausible* failure vs *visible*
   failure.
3. **Multi-condition fine-tuning (MCT-FT) specifically targets the hallucination
   failure mode — replicated across three architectures.** Phase 6 is a full
   3 × 3 grid: Whisper-small (50k BPE encoder-decoder), wav2vec 2.0 large
   (32-char CTC), and ESPnet E-Branchformer (5k BPE encoder-decoder + CTC) ×
   {off-the-shelf, clean-FT, MCT-FT}. On `babble_-5db`, MCT-FT reduces WER by
   **71 %** vs off-the-shelf in Whisper, **30 %** in wav2vec 2.0, and **43 %**
   in E-Branchformer. For Whisper specifically, the insertion-rate
   (hallucination) signal drops 91 %; output length collapses from 1.82× the
   reference to 1.06×. Total cloud GPU cost: ~$8 across all six fine-tunes.

![WER vs SNR — all 9 arms (3 architectures × 3 variants)](reports/ft_ablation/wer_vs_snr.png)

The `clean-FT` control isolates the contribution: in two of the three
architectures (Whisper, wav2vec 2.0), in-domain fine-tuning *alone* made
robustness *worse* than off-the-shelf on the catastrophic conditions
(e.g. Whisper `cellphone_in_crowd_-5db`: 178.7 % → 257.6 %). Only
*noise-aware* training fixes it. The cross-arch consistency shows this is
a more general property of single-condition FT, not a Whisper-specific quirk.
See [§4 of REPORT.md](reports/REPORT.md#4--phase-6-multi-condition-fine-tuning-ablation)
for the full breakdown including paired transcripts and per-condition tables.

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
- **Multi-model benchmark**: Whisper, wav2vec 2.0, and an ESPnet
  E-Branchformer, each in both off-the-shelf and fine-tuned variants, all
  scored through one harness.

## Repository layout

```
configs/            Experiment + degradation + model + training configs (YAML)
src/asr_robustness/
  audio.py          Audio I/O and resampling
  degrade/          The degradation harness (effects + pipeline)
  data/             Dataset download + manifest construction
  models/           Model registry + decoding adapters
  eval/             Decoding runner + WER scoring + breakdowns
  train/            Fine-tuning drivers — Whisper / wav2vec 2.0 / ESPnet (Phase 6 ablations)
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

* **Phases 1–4** (scaffold, degradation harness, data acquisition, model
  registry + WER scoring) — **done**. 78 unit tests passing.
* **Phase 5 — pilot:** **done**. Whisper-base + wav2vec 2.0-base across 16
  conditions × 100 `dev-clean` utts. Artifacts in
  [`reports/pilot/`](reports/pilot/). Full `test-clean` sweep planned for v1.1.
* **Phase 6 — symmetric fine-tuning ablation:** **done**. 3 × 3 grid:
  `whisper-small` / `wav2vec2-large-960h` / `asapp/e_branchformer_librispeech`
  each in `off-the-shelf` / `clean-FT` / `MCT-FT` variants, evaluated
  head-to-head on 100 `test-clean` utts × 16 conditions. Artifacts in
  [`reports/ft_ablation/`](reports/ft_ablation/). Cloud GPU spend: ~$8 on
  RunPod L40S (six fine-tunes across three architectures).
* **Phase 7 — breakdowns + real-corpus validation:** **v1.x roadmap**.
  Per-speaker stability slices and the full `test-clean` eval are cheap v1.1
  additions; **VOiCES real-corpus validation** is the credibility-anchoring
  v1.2 extension; Common Voice accent breakdown is v1.3. See
  [§6 of REPORT.md](reports/REPORT.md#6--limitations-and-the-v2-roadmap) for
  the full roadmap and ROI ranking.
* **Phase 8 — writeup:** **done** — [reports/REPORT.md](reports/REPORT.md).

### Reproducing the Phase 6 fine-tunes

Training does not run on the Mac dev box (MPS lacks the kernels HF Trainer
needs for bf16 Whisper, and ESPnet's MPS support is incomplete). The
fine-tuning scripts are written for any single-CUDA-GPU host (≥ 24 GB VRAM,
Ampere or later for bf16). A one-shot setup script targeting RunPod is
included:

```bash
# On a freshly provisioned A100 40GB / L40S 48GB / similar:
git clone https://github.com/bruce2tech/asr-robustness.git Speech_Recognition
cd Speech_Recognition
export HF_TOKEN=hf_...                             # use `read -rs` to keep it out of shell history
bash scripts/cloud_setup_runpod.sh                 # installs deps, downloads ~19 GB, smoke-checks

# Whisper-small fine-tunes (~1-1.5 h each on L40S):
make ft-clean
make ft-mct

# wav2vec 2.0 large fine-tunes (~1-1.5 h each on L40S):
make ft-wav2vec2-clean
make ft-wav2vec2-mct

# ESPnet E-Branchformer fine-tunes — needs pre-rendered data first
# (ESPnet's native dataloader has no in-process augmentation hook):
make prerender-espnet-train-clean
make prerender-espnet-train-mct
make prerender-espnet-dev
make ft-espnet-clean                                # ~1-1.5 h on L40S
make ft-espnet-mct                                  # ~1-1.5 h on L40S

# scp results/checkpoints/{whisper-small,wav2vec2-large,espnet-ebranchformer}-{clean,mct}-ft/
# back to the Mac, then locally:
make ft-eval                                        # 9-arm head-to-head (~7 h on M2 Max — ESPnet is CPU-bound)
make ft-report                                      # rebuild figures + summary tables
```

Total cloud spend on RunPod: ~$8 (six fine-tunes × ~1.5 h × $0.86/hr).

---

*Tooling note: this project was built with Claude Code as a coding assistant.
Research questions, experimental design, condition selection, ablation framing,
and interpretation of results were author-driven; the AI was used for
implementation and for surfacing trade-offs to choose between.*
