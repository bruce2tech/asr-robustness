# ASR Robustness Under Acoustic Degradation
# One-command reproduction targets. See README.md for details.

PY := speech_recognition/bin/python
PIP := speech_recognition/bin/pip

.PHONY: help setup test lint data data-minimal data-train data-full manifests demo pilot decode sweep ft-clean ft-mct ft-wav2vec2-clean ft-wav2vec2-mct prerender-espnet-train-clean prerender-espnet-train-mct prerender-espnet-dev ft-espnet-clean ft-espnet-mct ft-eval report ft-report clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dependencies (creates the venv if missing)
	@test -d speech_recognition || python3 -m venv speech_recognition
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:  ## Run the unit test suite
	$(PY) -m pytest

lint:  ## Lint the source tree
	$(PY) -m ruff check src tests

data:  ## Download datasets: LibriSpeech eval splits + train-clean-100 + MUSAN + RIRs (~80 GB)
	$(PY) -m asr_robustness.data.download --datasets librispeech musan rirs \
		--subsets dev-clean dev-other test-clean test-other train-clean-100

data-minimal:  ## Smoke-test download: LibriSpeech dev-clean only (~340 MB)
	$(PY) -m asr_robustness.data.download --datasets librispeech --minimal

data-train:  ## Slim Phase 6 cloud download: train-clean-100 + dev-clean + MUSAN + RIRs (~19 GB)
	$(PY) -m asr_robustness.data.download --datasets librispeech musan rirs \
		--subsets dev-clean train-clean-100

data-full:  ## Everything, incl. train-clean-360 (~100 GB+)
	$(PY) -m asr_robustness.data.download --datasets librispeech musan rirs

manifests:  ## Build JSON-lines manifests from downloaded LibriSpeech splits
	$(PY) -m asr_robustness.data.manifest --split dev-clean
	$(PY) -m asr_robustness.data.manifest --split test-clean
	$(PY) -m asr_robustness.data.manifest --split test-other
	$(PY) -m asr_robustness.data.manifest --split train-clean-100

demo:  ## Render clean vs degraded audio for one dev-clean utterance into demo/ (listen by ear)
	$(PY) -m asr_robustness.degrade.audition \
		--manifest manifests/librispeech_dev-clean.jsonl --index 0

pilot:  ## Pilot sweep: whisper-base over 100 utts x 12 conditions (~20-30 min)
	$(PY) -m asr_robustness.eval.run --config configs/experiments/pilot.yaml

decode:  ## Run all registered models over all conditions (Phase 4-5)
	$(PY) -m asr_robustness.eval.run --config configs/experiments/main.yaml

sweep:  ## Run the SNR degradation sweep (Phase 5)
	$(PY) -m asr_robustness.eval.run --config configs/experiments/snr_sweep.yaml

ft-clean:  ## Phase 6: fine-tune whisper-small on clean train-clean-100 (single-GPU cloud)
	$(PY) -m asr_robustness.train.whisper_ft --config configs/train/clean_ft.yaml

ft-mct:  ## Phase 6: fine-tune whisper-small with multi-condition (noise+reverb+codec) augmentation
	$(PY) -m asr_robustness.train.whisper_ft --config configs/train/mct_ft.yaml

ft-wav2vec2-clean:  ## Phase 6: fine-tune wav2vec2-base on clean train-clean-100 (single-GPU cloud)
	$(PY) -m asr_robustness.train.wav2vec2_ft --config configs/train/wav2vec2_clean_ft.yaml

ft-wav2vec2-mct:  ## Phase 6: fine-tune wav2vec2-base with multi-condition augmentation
	$(PY) -m asr_robustness.train.wav2vec2_ft --config configs/train/wav2vec2_mct_ft.yaml

prerender-espnet-train-clean:  ## Phase 6 ESPnet: render clean train-clean-100 to ESPnet Kaldi-style layout
	$(PY) -m asr_robustness.train.espnet_render_data \
		--manifest manifests/librispeech_train-clean-100.jsonl \
		--out-dir data/espnet/train-clean-100-clean --mode clean

prerender-espnet-train-mct:  ## Phase 6 ESPnet: render MCT train-clean-100 (~25 GB of degraded WAVs)
	$(PY) -m asr_robustness.train.espnet_render_data \
		--manifest manifests/librispeech_train-clean-100.jsonl \
		--out-dir data/espnet/train-clean-100-mct --mode mct \
		--conditions configs/train/espnet_mct_conditions.yaml \
		--noise-bank data/musan --rir-bank data/RIRS_NOISES/simulated_rirs

prerender-espnet-dev:  ## Phase 6 ESPnet: render clean dev-clean for eval (shared by both ablation arms)
	$(PY) -m asr_robustness.train.espnet_render_data \
		--manifest manifests/librispeech_dev-clean.jsonl \
		--out-dir data/espnet/dev-clean-clean --mode clean

ft-espnet-clean:  ## Phase 6 ESPnet: fine-tune asapp/e_branchformer_librispeech on clean data
	$(PY) -m asr_robustness.train.espnet_ft --config configs/train/espnet_clean_ft.yaml

ft-espnet-mct:  ## Phase 6 ESPnet: fine-tune with multi-condition (pre-rendered) augmentation
	$(PY) -m asr_robustness.train.espnet_ft --config configs/train/espnet_mct_ft.yaml

ft-eval:  ## Phase 6: evaluate the off-the-shelf / clean-FT / MCT-FT checkpoints head-to-head
	$(PY) -m asr_robustness.eval.run --config configs/experiments/ft_ablation.yaml

report:  ## Build report (plots + tables + summary.md) from results/snr_sweep.jsonl
	$(PY) -m asr_robustness.report.build

ft-report:  ## Build report from the Phase 6 9-arm ablation (results/ft_ablation.jsonl -> reports/ft_ablation/)
	$(PY) -m asr_robustness.report.build \
		--results results/ft_ablation.jsonl \
		--out reports/ft_ablation

clean:  ## Remove caches and build artifacts (keeps data/ and results/)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
