"""Endpoint tests. The OpenAI client is replaced by a fake, so no key or network is needed."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.classifier import Classifier
from app.main import app, get_classifier, get_threshold


class FakeOpenAI:
    """Mimics client.chat.completions.create and returns a canned JSON reply."""

    def __init__(self, content: str):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def reply(category="invoice", confidence=0.92, order_number="PO-48213", reason="ok"):
    return json.dumps({"category": category, "confidence": confidence, "order_number": order_number, "reason": reason})


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def use_reply(content: str, threshold: float = 0.75):
    fake = FakeOpenAI(content)
    app.dependency_overrides[get_classifier] = lambda: Classifier(client=fake, model="test-model", system_prompt="s")
    app.dependency_overrides[get_threshold] = lambda: threshold
    return fake


def test_confident_invoice_is_routed_to_invoices_sheet(client):
    use_reply(reply("invoice", 0.92, "PO-48213"))
    response = client.post("/classify", json={"email_id": "e1", "subject": "Invoice", "body": "attached", "sender": "a@x.example"})
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "email_id": "e1",
        "category": "invoice",
        "confidence": 0.92,
        "order_number": "PO-48213",
        "reason": "ok",
        "needs_review": False,
        "route": "invoices_sheet",
    }


def test_low_confidence_email_is_flagged_for_review(client):
    use_reply(reply("delay", 0.5, None))
    response = client.post("/classify", json={"email_id": "e2", "subject": "Hi", "body": "maybe late"})
    assert response.json()["needs_review"] is True
    assert response.json()["route"] == "review_sheet"


def test_unparseable_model_reply_goes_to_review_with_zero_confidence(client):
    use_reply("this is not json")
    response = client.post("/classify", json={"email_id": "e3", "subject": "x", "body": "y"})
    body = response.json()
    assert body["category"] == "other"
    assert body["confidence"] == 0.0
    assert body["route"] == "review_sheet"
    assert body["reason"].startswith("parse error")


def test_request_uses_structured_output_schema_and_model(client):
    fake = use_reply(reply())
    client.post("/classify", json={"email_id": "e4", "subject": "s", "body": "b", "sender": "f@x.example"})
    call = fake.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["messages"][0]["role"] == "system"
    assert "From: f@x.example" in call["messages"][1]["content"]


def test_missing_email_id_is_rejected(client):
    use_reply(reply())
    response = client.post("/classify", json={"subject": "s", "body": "b"})
    assert response.status_code == 422


def test_without_api_key_the_endpoint_says_so(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_classifier.cache_clear()
    response = client.post("/classify", json={"email_id": "e5", "subject": "s", "body": "b"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
