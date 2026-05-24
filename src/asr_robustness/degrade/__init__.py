"""The acoustic degradation harness.

`effects` holds the degradation primitives; `pipeline` composes them into named,
reproducible conditions driven by ``configs/degradation.yaml``.
"""

from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline, load_conditions

__all__ = ["NoiseBank", "RIRBank", "DegradationPipeline", "load_conditions"]
