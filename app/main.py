"""FastAPI service: POST /classify takes one email and returns its category, confidence, order number and route."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.classifier import Classifier
from app.routing import DEFAULT_CONFIDENCE_THRESHOLD, needs_review, route_for

app = FastAPI(title="Supplier email triage")


class EmailIn(BaseModel):
    email_id: str = Field(min_length=1, max_length=200)
    subject: str = Field(default="", max_length=998)
    body: str = Field(default="", max_length=100_000)
    sender: str = Field(default="", max_length=320)


class ClassifyOut(BaseModel):
    email_id: str
    category: str
    confidence: float
    order_number: str | None
    reason: str
    needs_review: bool
    route: str


@lru_cache
def get_classifier() -> Classifier:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not set")
    from openai import OpenAI  # imported here so tests do not need a key or network

    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    # LLM_BASE_URL points the OpenAI-compatible client at a provider; empty means OpenAI itself.
    base_url = os.environ.get("LLM_BASE_URL") or None
    return Classifier(client=OpenAI(api_key=api_key, base_url=base_url), model=model)


def get_threshold() -> float:
    raw = os.environ.get("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
    return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyOut)
def classify(
    email: EmailIn,
    classifier: Classifier = Depends(get_classifier),
    threshold: float = Depends(get_threshold),
) -> ClassifyOut:
    result = classifier.classify(email.subject, email.body, email.sender)
    return ClassifyOut(
        email_id=email.email_id,
        category=result.category,
        confidence=result.confidence,
        order_number=result.order_number,
        reason=result.reason,
        needs_review=needs_review(result, threshold),
        route=route_for(result, threshold),
    )
