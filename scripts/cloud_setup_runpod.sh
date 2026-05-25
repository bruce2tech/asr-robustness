#!/usr/bin/env bash
# One-shot setup for a RunPod (or any single-CUDA-GPU) box that brings the
# Phase 6 ASR Robustness fine-tuning runs to the "ready to train" point.
#
# Recommended pod template: any "PyTorch 2.x + CUDA 12.x" image with an
# A100 40GB / L40 48GB / A100 80GB GPU.
#
# Usage on the pod:
#
#   git clone <your-repo-url> /workspace/Speech_Recognition
#   cd /workspace/Speech_Recognition
#   export HF_TOKEN=hf_...                       # OR: run `hf auth login`
#   bash scripts/cloud_setup_runpod.sh
#
# After it finishes the next steps are:
#
#   make ft-clean                                # ~1-2 h on A100 40GB
#   make ft-mct                                  # ~1-2 h on A100 40GB
#
# When the runs are done, pull the checkpoints back to your Mac:
#
#   scp -r root@<pod-ip>:/workspace/Speech_Recognition/results/checkpoints \
#       /Users/patrickbruce/Documents/Speech_Recognition/results/
#
# then locally:
#
#   make ft-eval
#   make report
#
set -euo pipefail

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

step "Pre-flight: GPU + working directory"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. This script must run on a CUDA host (provision a GPU pod)."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [ "${GPU_MEM_MIB:-0}" -lt 22000 ]; then
    fail "GPU has <22 GB VRAM. whisper-small fine-tuning at batch 16 needs ~24 GB. \
Either pick a larger GPU or set gradient_checkpointing: true in configs/train/*.yaml."
elif [ "${GPU_MEM_MIB:-0}" -ge 40000 ]; then
    echo "(GPU has >=40 GB VRAM — safe to flip gradient_checkpointing: false for ~20-30% faster training.)"
fi

if [ ! -f pyproject.toml ] || [ ! -d src/asr_robustness ]; then
    fail "Run this from the repo root (the directory containing pyproject.toml)."
fi

step "Installing system dependencies (ffmpeg, tmux, rsync)"
# RunPod's PyTorch image has git + python + torch; we still need:
#   - ffmpeg for codec degradations (G.726, G.722, Opus in the harness)
#   - tmux so training survives SSH disconnects (long FT runs span ~5 h)
#   - rsync so the user can `rsync` checkpoints back to their Mac without
#     installing rsync mid-session (which is what we ended up doing on the
#     wav2vec 2.0 pod -- the install on-demand cost ~5 min of fumbling)
NEED_PKGS=""
for pkg in ffmpeg tmux rsync; do
    command -v "$pkg" >/dev/null 2>&1 || NEED_PKGS="$NEED_PKGS $pkg"
done
if [ -n "$NEED_PKGS" ]; then
    apt-get update -qq
    apt-get install -y -qq $NEED_PKGS
fi
ffmpeg -version | head -1
tmux -V

step "Pre-flight: GPU is unclaimed by zombie processes"
# A previous run may have left a Python process holding GPU memory after an
# abnormal exit (Ctrl-C mid-batch, SSH-drop SIGHUP). Kill any orphans before
# we try to allocate, otherwise the new run hits OutOfMemoryError immediately.
pkill -9 -f "asr_robustness.train" 2>/dev/null || true
pkill -9 -f "espnet2.bin.asr_train" 2>/dev/null || true
sleep 1
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
echo "GPU memory currently in use: ${GPU_USED} MiB (expect <100 after orphan cleanup)"

step "Creating Python venv and installing PyTorch (CUDA-matched, torch >= 2.6)"
# Mirror the local Mac layout (venv at speech_recognition/) so the existing
# Makefile targets work unchanged on the cloud box.
#
# Two requirements jointly constrain the PyTorch install:
#
# 1. The wheel's compiled CUDA toolkit must match the pod's driver. PyPI's
#    default torch wheel may target a newer CUDA than the driver supports,
#    triggering "The NVIDIA driver on your system is too old" at runtime.
#
# 2. torch must be >= 2.6 to avoid CVE-2025-32434 blocking `torch.load`.
#    transformers refuses to load legacy `pytorch_model.bin` checkpoints on
#    older torch, and -- worse -- refuses to resume from `optimizer.pt`
#    state files (which have no safetensors equivalent). On the wav2vec 2.0
#    pod we hit this twice and had to restart training from step 0 because
#    the optimizer state was unloadable.
#
# Both requirements are met by the cu124 wheels of torch >= 2.6: they target
# CUDA 12.4 (works on any driver supporting CUDA 12.4+, including all current
# RunPod L40S/A100/H100 templates) and include the CVE-2025-32434 fix.
test -d speech_recognition || python3 -m venv speech_recognition
speech_recognition/bin/pip install --upgrade pip
speech_recognition/bin/pip install "torch>=2.6" torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

step "Installing the rest of the project dependencies"
make setup

step "Checking HuggingFace auth"
# Auth lets the whisper-small download bypass the shared-IP rate limit. Either
# HF_TOKEN env var (preferred for non-interactive setup) or a prior
# `hf auth login` is enough.
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN env var present."
elif speech_recognition/bin/hf auth whoami >/dev/null 2>&1; then
    echo "Logged in as: $(speech_recognition/bin/hf auth whoami | head -1)"
else
    cat <<'EOF'
WARNING: no HuggingFace auth detected.

  Anonymous downloads of openai/whisper-small may rate-limit on a shared
  cloud IP and stall your training run. Recommended:

      export HF_TOKEN=hf_...           # paste your token, then re-run this script
                       OR
      speech_recognition/bin/hf auth login   # interactive

  Continuing anyway — if the model download fails with a 429, set HF_TOKEN
  and re-run `make ft-clean`.

EOF
fi

step "Downloading training data (~19 GB)"
# train-clean-100 + dev-clean + MUSAN (full) + OpenSLR RIRs (full). Excludes
# test-clean / test-other / train-clean-360 -- those are only needed for the
# *local* ft-eval step that runs on the Mac after training.
make data-train
# Drop the compressed archives now that extraction is complete -- they
# duplicate the extracted data and double the volume footprint to ~43 GB
# (we hit the 60 GB volume cap on the first wav2vec2-large FT run because
# of this). Re-runs of data-train fetch only what's missing, so deleting
# _archives/ is safe.
rm -rf data/_archives

step "Building manifests"
speech_recognition/bin/python -m asr_robustness.data.manifest --split train-clean-100
speech_recognition/bin/python -m asr_robustness.data.manifest --split dev-clean

step "Smoke-checking the training stack"
# 30-second confirmation that torch sees the GPU, the harness imports cleanly,
# and the training dataset can produce one batch under augmentation. This
# catches almost every "config/path is wrong" class of failure before you
# commit an hour of GPU time.
speech_recognition/bin/python <<'PY'
import torch
from transformers import WhisperProcessor
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.train.dataset import WhisperFTDataset, prepare_processor

assert torch.cuda.is_available(), "torch can't see a CUDA GPU"
print("CUDA OK:", torch.cuda.get_device_name(0),
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

processor = prepare_processor(WhisperProcessor.from_pretrained("openai/whisper-small"))
pipeline = DegradationPipeline.from_config(
    "configs/degradation.yaml",
    noise_bank=NoiseBank.from_dir("data/musan"),
    rir_bank=RIRBank.from_dir("data/RIRS_NOISES/simulated_rirs"),
)
ds = WhisperFTDataset(
    "manifests/librispeech_train-clean-100.jsonl", processor,
    pipeline=pipeline,
    conditions=["clean", "noise_-5db", "babble_-5db", "cellphone_in_crowd_-5db"],
    limit=4,
)
for i in range(4):
    item = ds[i]
    print(f"  utt {i}: features={tuple(item['input_features'].shape)} "
          f"labels={len(item['labels'])}")
print("Smoke check OK.")
PY

cat <<EOF

==================================================================
Setup complete. Ready to train.

  make ft-clean        # ~1-2 h on A100 40GB -- the clean-FT baseline
  make ft-mct          # ~1-2 h on A100 40GB -- the multi-condition variant

When both finish, the checkpoints are in:
  results/checkpoints/whisper-small-clean-ft/
  results/checkpoints/whisper-small-mct-ft/

Pull them back to your Mac (run this on the Mac, not on the pod):
  scp -r root@<pod-ip>:/workspace/Speech_Recognition/results/checkpoints \\
      /Users/patrickbruce/Documents/Speech_Recognition/results/

Then locally:
  make ft-eval         # head-to-head decode through the eval harness
  make report          # build plots + summary.md

==================================================================
EOF
