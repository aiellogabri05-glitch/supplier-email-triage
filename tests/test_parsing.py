import json

import pytest

from app.schema import CATEGORIES, parse_classification


def reply(**overrides) -> str:
    data = {"category": "invoice", "confidence": 0.93, "order_number": "PO-48213", "reason": "Invoice attached."}
    data.update(overrides)
    return json.dumps(data)


def test_valid_reply_is_parsed():
    result = parse_classification(reply())
    assert result.category == "invoice"
    assert result.confidence == pytest.approx(0.93)
    assert result.order_number == "PO-48213"
    assert result.reason == "Invoice attached."


def test_all_five_categories_are_accepted():
    assert set(CATEGORIES) == {"invoice", "delay", "backorder", "quote", "other"}
    for category in CATEGORIES:
        assert parse_classification(reply(category=category)).category == category


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError, match="unknown category"):
        parse_classification(reply(category="complaint"))


def test_missing_category_is_rejected():
    with pytest.raises(ValueError, match="unknown category"):
        parse_classification(json.dumps({"confidence": 0.5, "order_number": None, "reason": ""}))


@pytest.mark.parametrize("confidence", [-0.1, 1.01, "0.9", None, True])
def test_bad_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        parse_classification(reply(confidence=confidence))


def test_confidence_bounds_are_inclusive():
    assert parse_classification(reply(confidence=0)).confidence == 0.0
    assert parse_classification(reply(confidence=1)).confidence == 1.0


def test_null_and_blank_order_number_become_none():
    assert parse_classification(reply(order_number=None)).order_number is None
    assert parse_classification(reply(order_number="   ")).order_number is None


def test_order_number_is_stripped():
    assert parse_classification(reply(order_number=" 77120 ")).order_number == "77120"


def test_non_string_order_number_is_rejected():
    with pytest.raises(ValueError, match="order_number"):
        parse_classification(reply(order_number=77120))


def test_non_json_is_rejected():
    with pytest.raises(json.JSONDecodeError):
        parse_classification("not json at all")


def test_json_that_is_not_an_object_is_rejected():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_classification("[1, 2, 3]")


def test_missing_reason_defaults_to_empty():
    data = {"category": "quote", "confidence": 0.8, "order_number": None}
    assert parse_classification(json.dumps(data)).reason == ""
