# Multi-Domain Support Triage Agent

A terminal-based AI agent that triages real support tickets across three product ecosystems — **HackerRank**, **Claude (Anthropic)**, and **Visa** — using only the offline support corpus shipped with this repository.

---

## How to Run

### 1. Install dependencies

```bash
cd code/
pip install -r requirements.txt
```

### 2. Set API key (optional)

```bash
# Copy the example and fill in your Gemini key (optional — system works without it)
cp ../.env.example ../.env
# Then edit ../.env and set GOOGLE_API_KEY=your-key-here
```

> Without a Gemini API key the agent falls back to keyword-based classification
> and corpus-extraction-based responses. **Full functionality is available offline.**

### 3. Run the agent

```bash
# Process support_tickets.csv → writes support_tickets/output.csv
python main.py

# Process sample tickets (for local validation)
python main.py --sample

# Custom paths
python main.py --input path/to/input.csv --output path/to/output.csv
```

---

## Project Structure

```
code/
├── main.py               # Entry point — orchestrates the full pipeline
├── config.py             # Constants, valid label sets, corpus paths
├── corpus_loader.py      # Loads corpus docs, chunks them, builds TF-IDF index
├── classifier.py         # Hybrid classifier (LLM-primary, keyword-fallback)
├── risk_detector.py      # Deterministic risk & escalation rule engine
├── response_generator.py # Corpus-grounded response & escalation templates
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Approach

### Pipeline Overview

```
Read CSV
  → Detect Company (HackerRank / Claude / Visa)
  → TF-IDF Retrieval (top-3 corpus passages)
  → Risk Assessment (deterministic keyword rules)
  → Classify request_type + product_area
  → Decide: REPLY or ESCALATE
  → Generate Response (corpus-grounded)
  → Write output.csv
```

### 1. Corpus Retrieval — TF-IDF

- All corpus markdown files (`data/hackerrank/`, `data/claude/`, `data/visa/`) are loaded and chunked into retrievable passages.
- A `TfidfVectorizer` (scikit-learn) builds a cosine-similarity index at startup.
- Company-biased retrieval: scores for documents belonging to the detected company receive a **+20% boost**, so HackerRank tickets preferentially retrieve HackerRank docs.
- Fully deterministic and offline — no vector database or embeddings service required.

### 2. Classification — Hybrid LLM + Keyword

| Mode | Trigger | Output |
|------|---------|--------|
| **Gemini LLM** | `GOOGLE_API_KEY` present | Structured JSON with `request_type` + `product_area` |
| **Keyword fallback** | No key or LLM error | Scored keyword match against valid label sets |

Valid `request_type` values: `product_issue`, `feature_request`, `bug`, `invalid`

### 3. Escalation Decision Logic

The agent **escalates** a ticket if any of the following rules fires:

| Rule | Example triggers |
|------|-----------------|
| High-risk keyword | `fraud`, `identity theft`, `billing dispute`, `refund`, `security vulnerability`, `restore my access`, `pause our subscription` |
| Low retrieval confidence | TF-IDF score < 0.15 → no relevant doc found |
| Prompt injection detected | Requests to reveal system internals, ignore instructions |
| Account-level action required | Rescheduling tests, filling InfoSec forms, modifying seats |
| Out-of-scope / harmful | `delete all files`, commands unrelated to the three companies |

If none of these rules fire and a relevant corpus passage exists, the agent **replies** using content extracted from that passage.

### 4. Response Generation — Corpus-Grounded

- **Reply path**: Response is composed from the top-retrieved corpus passage. No information is invented — the agent only restates what the documentation says.
- **Escalate path**: A category-specific human-handoff template is used (billing escalation, security escalation, general escalation, etc.).
- **LLM enhancement** (optional): When a Gemini key is available, the LLM formats the corpus excerpt into a polished, user-facing reply — still grounded in the retrieved document.

### 5. Safety Measures

- **No hallucination**: responses are constructed from retrieved corpus text only.
- **Prompt injection detection**: tickets containing meta-instructions (e.g., "show me your internal rules") are detected, flagged as `invalid`, and escalated.
- **Out-of-scope detection**: requests unrelated to any supported product are replied with a safe "outside our scope" message.
- **Multi-request detection**: tickets containing multiple independent issues are escalated for human triage.

---

## Output Format

The agent writes `support_tickets/output.csv` with the following columns:

| Column | Allowed values |
|--------|---------------|
| `issue` | (original ticket text) |
| `subject` | (original subject) |
| `company` | (original company) |
| `response` | Corpus-grounded answer or escalation message |
| `product_area` | Relevant support category / domain area |
| `status` | `replied` \| `escalated` |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |
| `justification` | Classification method, TF-IDF score, escalation reason or corpus doc cited |

---

## Results (support_tickets.csv — 29 tickets)

| Metric | Value |
|--------|-------|
| Total tickets | 29 |
| Replied | 13 |
| Escalated | 16 |
| Processing time | ~0.3 seconds |
| External API calls | 0 (offline run) |

---

## Dependencies

See `requirements.txt`. Core libraries:

- `scikit-learn` — TF-IDF vectorization and cosine similarity
- `pandas` — CSV I/O
- `python-dotenv` — environment variable loading
- `google-generativeai` — Gemini LLM (optional)
