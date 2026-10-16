"""Measure classification accuracy on the labelled test emails, using the real OpenAI API.

Needs OPENAI_API_KEY in the environment. Costs a few cents per run. Prints the sample size,
category accuracy, order-number accuracy, and how many emails the review rule sent to review.

    python scripts/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from app.classifier import Classifier
from app.routing import DEFAULT_CONFIDENCE_THRESHOLD, needs_review

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labelled_emails.json"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set; nothing was measured.")
    from openai import OpenAI

    classifier = Classifier(client=OpenAI(api_key=api_key), model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))
    emails = json.loads(FIXTURE.read_text(encoding="utf-8"))

    category_hits = order_hits = order_expected = review_count = 0
    confusion: Counter[tuple[str, str]] = Counter()
    started = time.perf_counter()
    for email in emails:
        result = classifier.classify(email["subject"], email["body"], email["sender"])
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
              f"conf={result.confidence:.2f} order={result.order_number} review={needs_review(result, threshold)}")

    n = len(emails)
    elapsed = time.perf_counter() - started
    print(f"\nsample size: {n} labelled emails, model: {classifier._model}, threshold: {threshold}")
    print(f"category accuracy: {category_hits}/{n} = {category_hits / n:.1%}")
    print(f"order-number accuracy: {order_hits}/{order_expected} emails that have an order number")
    print(f"sent to review: {review_count}/{n}")
    print(f"mean seconds per email (incl. API latency): {elapsed / n:.2f}")
    if confusion:
        print("confusions (expected -> got): " + ", ".join(f"{a}->{b} x{c}" for (a, b), c in confusion.items()))


if __name__ == "__main__":
    main()
