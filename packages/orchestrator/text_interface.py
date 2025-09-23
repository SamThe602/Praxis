"""Natural language text interface orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from packages.translator import confidence as confidence_module
from packages.translator.decoder import TranslationResult, decode


@dataclass(slots=True)
class TranslationOutcome:
    """Wrapper carrying translation results or a fallback form."""

    prompt: str
    locale: str
    status: str
    confidence: float
    translation: TranslationResult | None
    fallback: Mapping[str, Any] | None


def translate_prompt(
    prompt: str,
    *,
    locale: str = "en",
    config: Mapping[str, Any] | None = None,
) -> TranslationOutcome:
    """Translate ``prompt`` into a structured specification or request form fill.

    The confidence threshold defaults to :data:`packages.translator.confidence.DEFAULT_THRESHOLD`.
    We keep the comment here to document the gating strategy: staying conservative
    reduces the risk of feeding low-quality specs into the synthesis pipeline.
    """

    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    config = dict(config or {})
    decoder_config = config.get("decoder", {})
    threshold = float(config.get("confidence_threshold", confidence_module.DEFAULT_THRESHOLD))

    if not 0.0 < threshold <= 1.0:
        raise ValueError("confidence_threshold must be within (0, 1]")

    translation = decode(prompt, locale=locale, config=decoder_config)
    confidence = translation.confidence

    if confidence >= threshold:
        status = "auto_accepted"
        fallback: Mapping[str, Any] | None = None
    else:
        status = "needs_confirmation"
        fallback = _build_form_fill_template(translation)

    return TranslationOutcome(
        prompt=prompt,
        locale=locale,
        status=status,
        confidence=confidence,
        translation=translation,
        fallback=fallback,
    )


def _build_form_fill_template(result: TranslationResult) -> Mapping[str, Any]:
    spec = result.spec
    required_fields = ["id", "inputs", "outputs", "constraints"]
    pending = [field for field in required_fields if not spec.get(field)]
    return {
        "mode": "form_fill",
        "prompt": result.prompt,
        "prefill": {field: spec.get(field) for field in required_fields},
        "required_fields": required_fields,
        "pending_fields": pending,
        "yaml_preview": result.yaml,
        "confidence": result.confidence,
    }
