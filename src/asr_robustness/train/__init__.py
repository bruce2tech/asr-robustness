"""Whisper fine-tuning (Phase 6).

The training-side counterpart to :mod:`asr_robustness.eval`. The thesis of this
phase is that **multi-condition (noise-augmented) fine-tuning flattens the
WER-vs-SNR curve** vs. an identical fine-tune on clean audio. Both ablation
runs (clean-FT, MCT-FT) share this module; they differ only in whether the
:class:`WhisperFTDataset` is given a degradation pipeline.

Training itself runs on a single CUDA GPU off-box (see the README "Cloud GPU
launch" section). The local Mac box is used only to author the code, prepare
manifests, and re-evaluate the fine-tuned checkpoints through the existing
:mod:`asr_robustness.eval` harness.
"""
