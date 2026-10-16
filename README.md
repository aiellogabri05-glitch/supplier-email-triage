# Supplier email triage: n8n + FastAPI + OpenAI

Supplier emails arrive in one test inbox. An n8n workflow reads each one, sends it to a FastAPI endpoint that classifies it with the OpenAI API (structured output), and routes it: invoices and quotes to Google Sheets, delays and backorders to Slack, and low-confidence or uncategorised emails to a review sheet. Every decision is logged as a row.

Use only test emails sent between your own accounts. Never send messages to real suppliers.

## Setup

1. Install Docker Desktop, then `cp .env.example .env` and set `OPENAI_API_KEY`.
2. `docker compose up -d --build` starts the API on http://localhost:8000 and n8n on http://localhost:5678.
3. Check the API: `curl localhost:8000/health`, then classify one email:
   `curl -X POST localhost:8000/classify -H "Content-Type: application/json" -d '{"email_id":"t1","subject":"Invoice INV-1 for PO-48213","body":"Attached.","sender":"billing@supplier.example"}'`
4. In n8n, create credentials: IMAP (test mailbox), Google Sheets (test sheet), Slack (test workspace).
5. Create one Google Sheet with tabs `Invoices`, `Quotes`, `Review` and `Decision log`. The `Decision log` tab needs the header row `logged_at, email_id, category, confidence, route`.
6. In n8n, use **Import from file** and choose `n8n/workflow.json`.
7. Open each node that shows a credential or ID placeholder (`REPLACE_WITH_...`) and select your credential, sheet, and Slack channel.
8. Set `TRIAGE_API_URL` for n8n to `http://api:8000` (already set in docker-compose.yml).
9. Send three test emails from a second account to the test inbox, then run the workflow.
10. Run the tests: `pip install -r requirements-dev.txt && python -m pytest`.

## Classification

- Categories: `invoice`, `delay`, `backorder`, `quote`, plus the fallback `other`.
- `POST /classify` returns `category`, `confidence` (0 to 1), `order_number` (copied from the text, or null), `reason`, `needs_review` and `route`.
- The prompt is in `prompts/classify_system.txt`. The JSON schema is in `app/schema.py`.
- Replies that fail validation become `other` with confidence 0, so they go to review.

## Routing rule

An email goes to review when its category is `other` or its confidence is below `CONFIDENCE_THRESHOLD` (default 0.75). Otherwise it goes to its category route. The rule is in `app/routing.py` and is covered by `tests/test_routing.py`.

## Files

- `app/main.py`: FastAPI app and `/classify`.
- `app/classifier.py`: OpenAI call, injected client for tests.
- `app/schema.py`: category list, schema, and reply parsing.
- `app/routing.py`: confidence rule and route keys.
- `n8n/workflow.json`: exported n8n workflow (IMAP trigger, HTTP call, Switch, Sheets/Slack, log).
- `scripts/run_eval.py`: accuracy measurement on `tests/fixtures/labelled_emails.json` (needs a key).
- `tests/`: parsing, routing and endpoint tests. The OpenAI client is faked.

## Known limits

- The labelled test set is 15 synthetic emails written for this project, not real supplier mail. Accuracy on it says little about real mail.
- The workflow JSON has not been imported or run in n8n. Node versions and field names may need small fixes when you import it.
- No retry or error branch if the API call fails. Such an execution just shows as failed in n8n.
- The confidence score is the model's own self-report, not a calibrated probability.
