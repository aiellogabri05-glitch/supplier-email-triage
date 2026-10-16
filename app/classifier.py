"""Call the OpenAI API for one email, using structured output, and parse the reply."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.schema import CLASSIFICATION_SCHEMA, Classification, parse_classification

log = logging.getLogger("triage.classifier")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classify_system.txt"
MAX_BODY_CHARS = 8000
SCHEMA_NAME = "supplier_email_classification"


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_user_message(subject: str, body: str, sender: str) -> str:
    body = body[:MAX_BODY_CHARS]
    return f"From: {sender}\nSubject: {subject}\n\nBody:\n{body}"


class Classifier:
    def __init__(self, client: Any, model: str, system_prompt: str | None = None) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt if system_prompt is not None else load_system_prompt()

    def classify(self, subject: str, body: str, sender: str = "") -> Classification:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": build_user_message(subject, body, sender)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": CLASSIFICATION_SCHEMA},
            },
        )
        raw = response.choices[0].message.content or ""
        try:
            return parse_classification(raw)
        except ValueError as exc:
            # A reply we cannot trust becomes "other" with zero confidence, so it goes to review.
            log.warning("unparseable classification reply: %s", exc)
            return Classification(category="other", confidence=0.0, order_number=None, reason=f"parse error: {exc}")
