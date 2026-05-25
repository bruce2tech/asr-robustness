"""CLI: fine-tune a wav2vec 2.0 CTC checkpoint, driven by a YAML config.

    python -m asr_robustness.train.wav2vec2_ft --config configs/train/wav2vec2_clean_ft.yaml

This is the wav2vec 2.0 arm of the Phase 6 ablation. Like the Whisper driver,
the same script trains both the clean-FT baseline and the multi-condition
(MCT) variant -- the only difference is whether the YAML config supplies an
``augmentation:`` block.

The wav2vec 2.0 fine-tuning recipe differs from Whisper's in three places:
* model: :class:`transformers.Wav2Vec2ForCTC` (not Seq2Seq)
* trainer: :class:`transformers.Trainer` (not Seq2SeqTrainer); CTC loss is the
  model's built-in forward loss, no generative decoding during training eval
* dataset: see :mod:`asr_robustness.train.wav2vec2_dataset`

Compute target is the same as the Whisper driver: any single CUDA GPU with
bf16 support (Ampere or later). The Mac dev box's MPS path lacks key kernels.

Output: the fine-tuned model + processor in ``output_dir``; the existing
:class:`asr_robustness.models.wav2vec2_hf.Wav2Vec2Model` adapter loads that
directory through HF ``from_pretrained`` like any model-hub ID.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jiwer
import numpy as np
import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.train.wav2vec2_dataset import (
    DataCollatorWav2Vec2FT,
    Wav2Vec2FTDataset,
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
        raise ValueError(
            f"augmentation conditions not defined in degradation config: {sorted(unknown)}"
        )
    return pipeline, conditions


def _make_compute_metrics(processor):
    """Return a Trainer-compatible compute_metrics that computes WER.

    Uses the same Whisper English text normalizer that the eval harness uses,
    so trainer-logged WER is directly comparable to the harness's WER numbers
    in ``reports/``. The CTC argmax decode is the standard inference path for
    wav2vec 2.0 base (no LM, no beam search).
    """
    normalizer = EnglishTextNormalizer({})

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        # Replace -100 in labels (pad positions) with the tokenizer pad id so
        # batch_decode handles them as padding rather than a literal token.
        label_ids = np.where(
            pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id
        )
        preds = processor.batch_decode(pred_ids)
        refs = processor.batch_decode(label_ids, group_tokens=False)
        preds = [normalizer(p) for p in preds]
        refs = [normalizer(r) for r in refs]
        kept = [(p, r) for p, r in zip(preds, refs) if r.strip()]
        if not kept:
            return {"wer": float("nan")}
        preds, refs = zip(*kept)
        return {"wer": float(jiwer.wer(list(refs), list(preds)))}

    return compute_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a wav2vec 2.0 CTC checkpoint.")
    ap.add_argument("--config", required=True, help="training YAML config")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model_id = cfg["model_id"]
    output_dir = cfg["output_dir"]

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    # use_safetensors=True forces the safe .safetensors weight file rather
    # than legacy pytorch_model.bin. transformers blocks .bin loading on
    # torch<2.6 due to CVE-2025-32434; safetensors loading is unaffected.
    model = Wav2Vec2ForCTC.from_pretrained(model_id, use_safetensors=True)
    # During continued fine-tuning the feature extractor (the conv stack on
    # top of the raw waveform) is normally frozen so we don't disturb the
    # SSL-pretrained low-level representation -- this is the canonical recipe.
    model.freeze_feature_encoder()

    pipeline, conditions = _build_pipeline(cfg.get("augmentation"))
    train_ds = Wav2Vec2FTDataset(
        cfg["train_manifest"], processor,
        pipeline=pipeline, conditions=conditions, limit=cfg.get("train_limit"),
    )
    eval_ds = None
    if cfg.get("eval_manifest"):
        # Eval is intentionally unaugmented so the trainer-logged WER reports
        # whether MCT preserves clean-set accuracy as training progresses.
        eval_ds = Wav2Vec2FTDataset(
            cfg["eval_manifest"], processor, limit=cfg.get("eval_limit", 200)
        )

    collator = DataCollatorWav2Vec2FT(processor=processor)

    t = cfg.get("training", {})
    use_bf16 = bool(t.get("bf16", False))
    use_fp16 = bool(t.get("fp16", torch.cuda.is_available() and not use_bf16))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=t.get("learning_rate", 1e-5),
        warmup_steps=t.get("warmup_steps", 500),
        max_steps=t.get("max_steps", 4000),
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=t.get("save_steps", 1000),
        save_total_limit=t.get("save_total_limit", 2),
        logging_steps=t.get("logging_steps", 50),
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=t.get("report_to", "none"),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=cfg.get("seed", 0),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(processor) if eval_ds is not None else None,
        processing_class=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nFinal fine-tuned model + processor saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
