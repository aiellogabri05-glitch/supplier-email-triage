"""Measure classification accuracy on the labelled test emails, using the configured LLM endpoint.

Needs LLM_API_KEY in the environment (see .env.example). Each email is retried on transient
errors; an email that still fails is reported and left out of the accuracy denominators.
Prints the sample size, category accuracy, order-number accuracy, and how many emails the
review rule sent to review.

    python scripts/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Run as `python scripts/run_eval.py` from anywhere: make the project root importable first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import Classifier  # noqa: E402
from app.routing import DEFAULT_CONFIDENCE_THRESHOLD, needs_review  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labelled_emails.json"
ATTEMPTS = 5
RETRY_WAIT_SECONDS = 20


def classify_with_retry(classifier: Classifier, email: dict):
    """Return (result, None) on success or (None, error text) after all attempts failed."""
    last_error = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return classifier.classify(email["subject"], email["body"], email["sender"]), None
        except Exception as exc:  # the endpoint returns transient 503s; keep going
            last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            print(f"  {email['email_id']} attempt {attempt}/{ATTEMPTS} failed: {last_error}", flush=True)
            if attempt < ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
    return None, last_error


def main() -> None:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        sys.exit("LLM_API_KEY is not set; nothing was measured.")
    from openai import OpenAI

    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    classifier = Classifier(
        client=OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=120),
        model=model,
    )
    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))
    emails = json.loads(FIXTURE.read_text(encoding="utf-8"))

    category_hits = order_hits = order_expected = review_count = 0
    measured = 0
    failed: list[str] = []
    confusion: Counter[tuple[str, str]] = Counter()
    started = time.perf_counter()
    for email in emails:
        result, error = classify_with_retry(classifier, email)
        if result is None:
            failed.append(f"{email['email_id']} ({error})")
            continue
        measured += 1
        if result.category == email["expected_category"]:
            category_hits += 1
        else:
            confusion[(email["expected_category"], result.category)] += 1
        if email["expected_order"] is not None:
            order_expected += 1
            if result.order_number == email["expected_order"]:
                order_hits += 1
        if needs_review(result, threshold):
            review_count += 1
        print(f"{email['email_id']} expected={email['expected_category']} got={result.category} "
              f"conf={result.confidence:.2f} order={result.order_number} review={needs_review(result, threshold)}",
              flush=True)

    total = len(emails)
    elapsed = time.perf_counter() - started
    print(f"\nlabelled emails in set: {total}; classified successfully: {measured}; failed: {len(failed)}")
    print(f"model: {model}; threshold: {threshold}")
    if measured:
        print(f"category accuracy: {category_hits}/{measured} = {category_hits / measured:.1%}")
        print(f"order-number accuracy: {order_hits}/{order_expected} emails that have an order number")
        print(f"sent to review: {review_count}/{measured}")
        print(f"mean seconds per classified email (incl. retries): {elapsed / measured:.2f}")
    if confusion:
        print("confusions (expected -> got): " + ", ".join(f"{a}->{b} x{c}" for (a, b), c in confusion.items()))
    if failed:
        print("failed emails: " + "; ".join(failed))


if __name__ == "__main__":
    main()
