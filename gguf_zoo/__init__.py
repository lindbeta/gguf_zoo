"""gguf-zoo: GGUF metadata -> deterministic fictional creature field guide."""
from __future__ import annotations

from .anomalies import anomaly_organs
from .attributes import (ATTR_LABELS, FICTIONAL_ATTRIBUTES, REAL_ATTRIBUTES,
                         classify_attribute, eval_attribute_classifier,
                         train_attribute_classifier)
from .classifier import (classify_input_type, classify_input_type_consensus,
                         eval_classifier, resolve_device, train_classifier)
from .parser import GGUFError, ModelMeta, build_meta, read_metadata
from .render import build_description, render_prompt
from .rules import (Features, Profile, QuantInfo, build_profile, estimate_params,
                    extract_features, quant_info, size_tier)
from .senses import (FICTIONAL_SENSES, INTERNAL_SENSES, VISIBLE_SENSES, Sense,
                     SenseResult, detect_input_types)
from .taxonomy import binomial, gloss

__all__ = [
    "read_metadata", "build_meta", "ModelMeta", "GGUFError",
    "extract_features", "estimate_params", "size_tier", "quant_info",
    "build_profile", "Features", "Profile", "QuantInfo",
    "classify_input_type", "classify_input_type_consensus", "detect_input_types",
    "train_classifier", "eval_classifier", "resolve_device",
    "Sense", "SenseResult", "VISIBLE_SENSES", "INTERNAL_SENSES", "FICTIONAL_SENSES",
    "binomial", "gloss", "anomaly_organs", "build_description", "render_prompt",
    "classify_attribute", "train_attribute_classifier", "eval_attribute_classifier",
    "ATTR_LABELS", "REAL_ATTRIBUTES", "FICTIONAL_ATTRIBUTES",
]
