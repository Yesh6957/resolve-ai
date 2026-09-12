# Resolve.AI — Apple Support Triage Agent

Resolve.AI is an end-to-end AI customer support pipeline built for the Hiver SDE Intern take-home assignment. It classifies incoming Twitter support messages for **@AppleSupport**, decides whether each one can be safely auto-handled or must be escalated to a human, and drafts a reply grounded in the brand's own historical resolutions via Retrieval-Augmented Generation (RAG).

The project has two parts:
- A **Flask API backend** (`main.py`) that runs the real classification, safety-check, RAG retrieval, and evaluation logic against processed CSV data.
- A **single-page dashboard** (`index.html`) that visualizes the evaluation results and includes a live pipeline simulator for demoing the triage logic in-browser.

**Live demo:** https://resolve-ai-7few.onrender.com/
> Hosted on Render's free tier — the server spins down when idle, so the first load can take 30–45 seconds.

## What it does

1. **Deterministic safety layer** — regex-based checks run before any LLM call to flag PII, explicit toxicity, and physical hardware damage keywords, guaranteeing escalation on obvious violations without depending on probabilistic LLM output.
2. **LLM intent classification** — an LLM classifies each tweet into one of 7 consolidated intents (`general_inquiry`, `software_troubleshooting`, `hardware_repair`, `billing_purchases`, `account_security`, `device_connectivity`, `other`) and returns a self-reported confidence score. Anything below a 0.60 confidence threshold is auto-escalated.
3. **ChromaDB RAG retrieval** — for messages deemed safe to auto-handle, the tweet is embedded and matched against a local ChromaDB store of historical Apple Support resolutions.
4. **Grounded reply generation** — a reply is drafted strictly from the retrieved historical examples. If a ticket is escalated, no reply is generated at all (see Decision Log) to avoid wasted tokens and unnecessary brand risk.
5. **Evaluation harness** — a golden set of 200 hand-labelled examples (kept strictly isolated from the RAG knowledge base) is scored against intent accuracy, escalation precision/recall, per-class precision/recall/F1, and LLM-as-a-Judge tone/grounding scores.

## Headline results

| Metric | Trivial Baseline | Simple Heuristic (Regex) | Agent (LLM) |
|---|---|---|---|
| Intent Accuracy | 60.00% | 44.50% | 28.50% |
| Escalation Accuracy | 52.50% | 56.00% | 58.50% |
| Escalation Precision | — | — | 67.65% |
| Escalation Recall | — | — | 24.21% |

**The headline escalation accuracy (58.50%) is misleading** — it's inflated by true negatives (benign tweets correctly left alone) and hides an escalation recall of only 24.21%, meaning the agent misses roughly 3 out of 4 messages that actually need a human. See the full report for the confusion matrix and why this matters more than the top-line number.

Full breakdown, per-intent metrics, failure analysis, and the 14-item decision log are in the report (linked below).

## Tech stack

- **Backend:** Python, Flask, Flask-CORS, Gunicorn
- **Data processing:** Pandas, NumPy, scikit-learn (metrics: precision/recall/F1, confusion matrices)
- **RAG:** ChromaDB (local vector store)
- **LLM:** Groq (Codestral) for intent classification and reply generation; Google Gemini (`google-genai`) for LLM-as-a-Judge scoring
- **Frontend:** HTML5, CSS3, vanilla JavaScript, Chart.js
- **Notebooks:** Jupyter, used for data cleaning and golden-set labelling

## Repo structure

```
resolve-ai/
├── main.py                     # Flask API — serves metrics, evaluation, and dashboard data
├── index.html                  # Dashboard + live pipeline simulator (single-page app)
├── data/
│   └── processed/
│       ├── golden_dataset.csv        # 200 hand-labelled examples (evaluation ground truth)
│       ├── evaluation_results.csv    # Agent predictions vs. ground truth per example
│       ├── rag_knowledge_base.csv    # ~1,800 historical resolutions used for RAG retrieval
│       └── baseline_results.txt      # Trivial + heuristic baseline outputs
├── requirements.txt
└── .env.example                 # API keys required (Groq, Gemini) — see Setup below
```
> ⚠️ Adjust this section to match your actual pipeline/ingestion/evaluation script names (e.g. `pipeline.py`, `evaluate.py`, `baselines.py`) — fill in whatever the real filenames are before submitting, since this is what you'll be asked to walk through live.

## Setup and reproduction

```bash
# 1. Clone and enter the repo
git clone https://github.com/Yesh6957/resolve-ai.git
cd resolve-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys
cp .env.example .env
# then fill in your GROQ_API_KEY and GEMINI_API_KEY

# 4. Run the backend
python main.py
# Flask server starts at http://localhost:5000

# 5. Open index.html in your browser (or serve it via Flask's static route)
```

To regenerate the golden-set evaluation and baseline numbers from scratch, run the pipeline and evaluation scripts against `data/processed/` — see inline comments in `main.py` for the expected CSV schema (`true_intent`, `pred_intent`, `true_escalate`, `pred_escalate`, `judge_tone`, `judge_grounding`).

## API endpoints (`main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /api/executive-summary` | Headline accuracy, escalation precision/recall/F1 |
| `GET /api/class-wise-performance` | Per-intent precision/recall/F1 |
| `GET /api/confusion-matrices` | Intent + escalation confusion matrices |
| `GET /api/safety-failures` | False-negative escalations (missed critical cases) |
| `GET /api/rag-evidence` | Sample RAG retrievals with similarity scores |
| `GET /api/llm-judge` | LLM-as-a-Judge tone/grounding scores |
| `GET /api/failure-cases` | Top intent-mismatch examples |
| `GET /api/intent-distribution` | Golden-set class distribution (for charts) |
| `GET /api/accuracy-comparison` | Baseline vs. agent accuracy (for charts) |
| `GET /api/misleading-metrics` | Explanation of why the headline number is misleading |
| `GET /api/decision-log` | Architecture decisions and rationale |
| `GET /api/future-improvements` | Planned next steps |

## Report

Full report — problem framing, baselines, per-intent metrics, 5 failure modes with real examples, the misleading-metrics analysis, and a 14-item decision log — is available here:
[Hiver AI Assignment Report](https://drive.google.com/file/d/1v0LRXOznDyZps75GSKUwK3qaBDU-sY1U/view?usp=drivesdk)

## About the developer

Built by Yeshwanth J., pursuing a Master of Computer Applications (expected 2026). This project was built end-to-end: extracting and cleaning a 3M-row noisy Twitter dataset, manually constructing a leakage-free golden evaluation set, building a hybrid deterministic + LLM triage pipeline, and implementing an automated evaluation harness with LLM-as-a-Judge scoring.

- [GitHub Repository](https://github.com/Yesh6957/resolve-ai)
- [Portfolio](https://yesh-portfolio-dev.vercel.app/)
-
