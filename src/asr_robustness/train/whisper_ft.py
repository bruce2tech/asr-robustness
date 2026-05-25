"""CLI: fine-tune a Whisper checkpoint, driven by a YAML config.

    python -m asr_robustness.train.whisper_ft --config configs/train/clean_ft.yaml

This is the Phase 6 ablation runner. The same script trains both the
**clean-FT** baseline and the **multi-condition (MCT)** variant -- the only
difference is whether the YAML config supplies an ``augmentation`` block.

Compute target: single CUDA GPU (Lambda Labs, RunPod, Modal, Colab Pro, ...).
The Mac dev box has no NVIDIA GPU; do not run this script locally on M-series
hardware (MPS lacks the kernels HF Trainer needs for FP16/BF16 Whisper).

The output directory holds the fine-tuned weights + tokenizer + feature
extractor. Point a ``models:`` entry at that directory (``model_id: path/to/dir``)
and the existing :class:`asr_robustness.models.whisper_hf.WhisperModel` adapter
will load it like any HuggingFace checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jiwer
import numpy as np
import torch
import yaml
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.train.dataset import (
    DataCollatorWhisperFT,
    WhisperFTDataset,
    prepare_processor,
)


def _maybe_bank(cls, path: str | None):
    if path and Path(path).is_dir():
        return cls.from_dir(path)
    return None


def _build_pipeline(aug_cfg: dict | None) -> tuple[DegradationPipeline | None, list[str] | None]:
    """Build a degradation pipeline + condition list from the YAML aug block."""
    if not aug_cfg:
        return None, None
    pipeline = DegradationPipeline.from_config(
        aug_cfg["degradation_config"],
        noise_bank=_maybe_bank(NoiseBank, aug_cfg.get("noise_bank")),
        rir_bank=_maybe_bank(RIRBank, aug_cfg.get("rir_bank")),
    )
    conditions = aug_cfg["conditions"]
    if not conditions:
        raise ValueError("augmentation.conditions must be non-empty")
    unknown = set(conditions) - set(pipeline.condition_names())
    if unknown:
        raise ValueError(f"augmentation conditions not defined in degradation config: {sorted(unknown)}")
    return pipeline, conditions


def _make_compute_metrics(processor):
    """Return a Trainer-compatible ``compute_metrics`` that computes WER.

    The Whisper English normalizer is the same normalizer used everywhere else
    in this project (see :mod:`asr_robustness.eval.scoring`), so the WER number
    that the trainer logs during eval is comparable to the harness's WER.
    """
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        preds = [normalizer(p) for p in preds]
        refs = [normalizer(r) for r in refs]
        # Drop pairs where the reference normalized to empty (the normalizer
        # nukes pure-punctuation lines): jiwer otherwise raises.
        kept = [(p, r) for p, r in zip(preds, refs) if r.strip()]
        if not kept:
            return {"wer": float("nan")}
        preds, refs = zip(*kept)
        return {"wer": float(jiwer.wer(list(refs), list(preds)))}

    return compute_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a Whisper checkpoint.")
    ap.add_argument("--config", required=True, help="training YAML config")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model_id = cfg["model_id"]
    language = cfg.get("language", "en")
    task = cfg.get("task", "transcribe")
    output_dir = cfg["output_dir"]

    processor = prepare_processor(
        WhisperProcessor.from_pretrained(model_id), language=language, task=task
    )
    # use_safetensors=True forces the safe .safetensors weight file rather
    # than legacy pytorch_model.bin. transformers blocks .bin loading on
    # torch<2.6 due to CVE-2025-32434; safetensors loading is unaffected.
    model = WhisperForConditionalGeneration.from_pretrained(model_id, use_safetensors=True)
    # During training, let the model insert language/task tokens itself.
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    pipeline, conditions = _build_pipeline(cfg.get("augmentation"))
    train_ds = WhisperFTDataset(
        cfg["train_manifest"], processor,
        pipeline=pipeline, conditions=conditions, limit=cfg.get("train_limit"),
    )
    eval_ds = None
    if cfg.get("eval_manifest"):
        # Eval is intentionally unaugmented: we want WER on the clean dev set
        # logged each step so we can see whether MCT preserves clean accuracy.
        eval_ds = WhisperFTDataset(
            cfg["eval_manifest"], processor, limit=cfg.get("eval_limit", 200)
        )

    collator = DataCollatorWhisperFT(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    train_args_cfg = cfg.get("training", {})
    use_bf16 = bool(train_args_cfg.get("bf16", False))
    use_fp16 = bool(train_args_cfg.get("fp16", torch.cuda.is_available() and not use_bf16))
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_args_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=train_args_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_args_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_args_cfg.get("learning_rate", 1e-5),
        warmup_steps=train_args_cfg.get("warmup_steps", 500),
        max_steps=train_args_cfg.get("max_steps", 4000),
        gradient_checkpointing=train_args_cfg.get("gradient_checkpointing", True),
        max_grad_norm=train_args_cfg.get("max_grad_norm", 1.0),
        fp16=use_fp16,
        bf16=use_bf16,
        predict_with_generate=True,
        generation_max_length=train_args_cfg.get("generation_max_length", 225),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=train_args_cfg.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=train_args_cfg.get("save_steps", 1000),
        save_total_limit=train_args_cfg.get("save_total_limit", 2),
        logging_steps=train_args_cfg.get("logging_steps", 50),
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=train_args_cfg.get("report_to", "none"),
        dataloader_num_workers=train_args_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=cfg.get("seed", 0),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(processor) if eval_ds is not None else None,
        processing_class=processor.feature_extractor,  # transformers>=5: renamed from `tokenizer=`. Lets Trainer save the processor alongside weights.
    )

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nFinal fine-tuned model + processor saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
