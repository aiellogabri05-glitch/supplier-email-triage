import pytest

from app.routing import CATEGORY_ROUTES, REVIEW_ROUTE, needs_review, route_for
from app.schema import Classification


def make(category="invoice", confidence=0.9) -> Classification:
    return Classification(category=category, confidence=confidence, order_number=None, reason="")


def test_confident_real_category_is_not_reviewed():
    assert not needs_review(make("invoice", 0.9), threshold=0.75)


def test_confidence_exactly_at_threshold_is_not_reviewed():
    assert not needs_review(make("delay", 0.75), threshold=0.75)


def test_confidence_just_below_threshold_is_reviewed():
    assert needs_review(make("delay", 0.7499), threshold=0.75)


def test_other_is_always_reviewed_even_at_full_confidence():
    assert needs_review(make("other", 1.0), threshold=0.75)


@pytest.mark.parametrize("category, route", list(CATEGORY_ROUTES.items()))
def test_confident_email_goes_to_its_category_route(category, route):
    assert route_for(make(category, 0.95), threshold=0.75) == route


def test_low_confidence_email_goes_to_review_route():
    assert route_for(make("quote", 0.4), threshold=0.75) == REVIEW_ROUTE


def test_other_goes_to_review_route():
    assert route_for(make("other", 0.99), threshold=0.75) == REVIEW_ROUTE


def test_every_real_category_has_a_route():
    assert set(CATEGORY_ROUTES) == {"invoice", "delay", "backorder", "quote"}
