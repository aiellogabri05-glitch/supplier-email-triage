"""Confidence routing rule: which emails go to n8n's review route and which go to a category route."""

from __future__ import annotations

from app.schema import Classification

DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Route keys returned to n8n. n8n's Switch node maps each key to Slack or a Google Sheet.
CATEGORY_ROUTES = {
    "invoice": "invoices_sheet",
    "quote": "quotes_sheet",
    "delay": "delays_slack",
    "backorder": "backorders_slack",
}
REVIEW_ROUTE = "review_sheet"


def needs_review(classification: Classification, threshold: float) -> bool:
    """Send to review when the model is unsure or the email fits no real category."""
    return classification.category == "other" or classification.confidence < threshold


def route_for(classification: Classification, threshold: float) -> str:
    if needs_review(classification, threshold):
        return REVIEW_ROUTE
    return CATEGORY_ROUTES[classification.category]
