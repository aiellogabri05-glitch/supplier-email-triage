"""Classification schema (sent to OpenAI as structured output) and response parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass

# Four real categories plus "other", the fallback for anything unclear.
CATEGORIES = ("invoice", "delay", "backorder", "quote", "other")

# Strict-mode JSON schema for OpenAI structured outputs. Every property is required,
# and additionalProperties is false, as strict mode requires. Range checks for confidence
# happen in parse_classification, because strict mode does not accept min/max keywords.
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "confidence": {"type": "number"},
        "order_number": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["category", "confidence", "order_number", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float
    order_number: str | None
    reason: str


def parse_classification(raw: str) -> Classification:
    """Parse and validate the model's JSON reply. Raises ValueError if anything is wrong."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("classification reply is not a JSON object")

    category = data.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence out of range 0..1: {confidence}")

    order_number = data.get("order_number")
    if order_number is not None and not isinstance(order_number, str):
        raise ValueError("order_number must be a string or null")
    order_number = order_number.strip() if order_number else None

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    return Classification(
        category=category,
        confidence=float(confidence),
        order_number=order_number or None,
        reason=reason.strip(),
    )
