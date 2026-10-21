# AI Resume Screening Platform — RAG Upgrade

This upgrade extends the original HPPCS02 capstone with a persistent vector store, semantic section chunking, candidate-level retrieval, an evaluation harness, FastAPI, and a GitHub Actions quality gate.

## What changed from the original project?
The original `main.py` uses rule-based skill/experience/education scoring plus cosine similarity. Sentence Transformers are an embedding model, not an LLM. This upgrade adds retrieval over resume chunks and returns evidence for each candidate. It does not claim to be a complete production system until evaluation labels, deployment security, monitoring, and a generation provider are configured.

## Setup (Python 3.10 or 3.11 recommended)

From the `Codebase` directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-rag.txt
python -m rag_system.ingest
```

The first run downloads the embedding model. Chroma's persistent data is written to `Codebase/chroma_db/`.

## Retrieve candidates

```python
from rag_system.pipeline import ResumeRAG
rag = ResumeRAG()
print(rag.retrieve("Find candidates with Kubernetes and Terraform experience", k=5))
```

The `score` is a semantic similarity signal, not a probability or a hiring recommendation. Results are deduplicated by resume ID, so one candidate cannot occupy multiple top-k slots.

## Run the API

```bash
uvicorn rag_system.api:app --host 127.0.0.1 --port 8000
```

- `GET /health`
- `POST /screen` with `{"query":"Candidates with AWS and Docker experience","top_k":5}`

The `/screen` response includes retrieved evidence and a prompt that can be passed to a configured LLM. No external LLM API is called by default.

## Evaluation — required before making performance claims

1. Review the actual 10 resumes and write human relevance labels for a diverse set of screening queries.
2. Add one JSON object per line to `eval_data.jsonl`:
   `{"query":"Which candidates have Kubernetes and Terraform?","relevant_ids":["cv2","cv7"]}`
3. IDs are resume filenames without `.pdf` (for example, `cv1`).
4. Run ingestion, then `python -m rag_system.evaluate`.

Metrics: Recall@5, Precision@5, MRR, and mean retrieval latency. Recall@5 quality gate defaults to 0.70 and can be overridden with `MIN_RECALL_AT_5`. Do not include unreviewed examples as ground truth. CI warns and skips the quality gate while no labels exist; once labeled cases are added, the gate runs automatically.

## Current limitations / next hardening steps

- Add a tested hybrid BM25 + vector retrieval fallback; `rank-bm25` is included as a dependency for that next step, but it is not silently presented as implemented.
- Add bounded retries/timeouts for the chosen LLM provider and JSON schema validation.
- Add structured logs, request IDs, token/latency/cost metrics, and error-rate monitoring.
- Add Docker deployment, non-root container user, secret management, TLS, authentication, rate limiting, and data retention controls before public exposure.
- OCR may be required for scanned/image-only PDFs.
- Validate experience extraction and skill aliases against real examples.

## Resume wording (only after you measure it)

Use a truthful interim description:
- Extended a resume-ranking capstone with semantic resume chunking, persistent vector retrieval, candidate-level evidence, and a Recall@5/Precision@5/MRR evaluation harness.
- Added a FastAPI screening endpoint and GitHub Actions quality checks for labeled retrieval evaluations.

Only add numerical improvements, latency figures, or “production deployed” after the system has been measured and deployed successfully.
