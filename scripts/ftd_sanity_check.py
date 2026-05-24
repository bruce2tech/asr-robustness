#!/usr/bin/env python3
"""Local sanity check for the Phase 6 fine-tuning dataset.

This does NOT train -- it just instantiates `WhisperFTDataset` against the real
train-clean-100 manifest and verifies that:

  1. The clean-FT branch (no pipeline) returns sensibly-shaped Whisper features
     and label sequences.
  2. The MCT-FT branch (pipeline + condition list) runs end-to-end -- audio
     load + degradation + feature extraction + tokenization.
  3. The two branches produce *different* features for the same utterance,
     confirming the augmentation actually had an effect.

If this works on the Mac, the same code path will work on the cloud GPU.
"""

from __future__ import annotations

import numpy as np
from transformers import WhisperProcessor

from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.train.dataset import WhisperFTDataset, prepare_processor

MANIFEST = "manifests/librispeech_train-clean-100.jsonl"
MODEL_ID = "openai/whisper-small"
CONDITIONS = ["clean", "noise_-5db", "babble_-5db", "cellphone_in_crowd_-5db", "reverb"]


def main() -> int:
    print(f"loading processor for {MODEL_ID} ...")
    processor = prepare_processor(WhisperProcessor.from_pretrained(MODEL_ID))

    # --- 1. clean-FT branch ---------------------------------------------------
    print("\n[clean-FT path] no pipeline -- features come from original audio")
    clean_ds = WhisperFTDataset(MANIFEST, processor, limit=3)
    for i in range(3):
        item = clean_ds[i]
        rec = clean_ds.records[i]
        print(
            f"  #{i} utt={rec['utt_id']:25s} "
            f"features={tuple(item['input_features'].shape)} "
            f"n_labels={len(item['labels'])}"
        )

    # --- 2. MCT-FT branch -----------------------------------------------------
    print("\n[MCT-FT path] building degradation pipeline ...")
    pipeline = DegradationPipeline.from_config(
        "configs/degradation.yaml",
        noise_bank=NoiseBank.from_dir("data/musan"),
        rir_bank=RIRBank.from_dir("data/RIRS_NOISES/simulated_rirs"),
    )
    print(f"  pipeline has {len(pipeline.condition_names())} conditions defined")
    print(f"  sampling from: {CONDITIONS}")

    mct_ds = WhisperFTDataset(
        MANIFEST, processor,
        pipeline=pipeline, conditions=CONDITIONS, limit=5,
    )
    for i in range(5):
        item = mct_ds[i]
        rec = mct_ds.records[i]
        print(
            f"  #{i} utt={rec['utt_id']:25s} "
            f"features={tuple(item['input_features'].shape)} "
            f"n_labels={len(item['labels'])}"
        )

    # --- 3. Confirm the augmentation actually changes the features -----------
    # Use a no-`clean` conditions list so every sample MUST be degraded, then
    # compare to the clean-FT features on the same indices.
    print("\n[augmentation effect check]")
    forced_aug_ds = WhisperFTDataset(
        MANIFEST, processor,
        pipeline=pipeline,
        conditions=["noise_-5db", "babble_-5db", "cellphone_in_crowd_-5db", "reverb"],
        limit=10,
    )
    deltas = []
    for i in range(10):
        clean_feat = np.asarray(clean_ds[i % len(clean_ds)]["input_features"])
        aug_feat = np.asarray(forced_aug_ds[i]["input_features"])
        deltas.append(float(np.abs(clean_feat - aug_feat).max()))
    augmented = sum(d > 1e-3 for d in deltas)
    print(f"  forced-augmentation conditions: {sorted(set(forced_aug_ds.conditions))}")
    print(f"  10 utts, max|delta| per item: {[round(d, 3) for d in deltas]}")
    print(f"  augmented (delta > 1e-3): {augmented}/10  "
          f"({'PASS' if augmented == 10 else 'FAIL -- some items not augmented'})")
    if augmented != 10:
        return 1
    print("\nsanity check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
