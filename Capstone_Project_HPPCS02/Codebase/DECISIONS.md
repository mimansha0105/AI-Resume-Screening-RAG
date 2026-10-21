# Architecture decisions

## Why ChromaDB?
Persistent local vector storage makes the first iteration simple to run and inspect. For a multi-user production deployment, configure a managed or PostgreSQL/pgvector-backed store and add access controls, backups, and lifecycle management.

## Why semantic section chunking?
Resume headings provide meaningful retrieval units such as Skills, Experience, and Projects. Oversized sections are split to bound chunk length. PDFs with unusual layouts may need parser-specific handling; inspect extracted text when retrieval is poor.

## Why keep BM25 as a future fallback?
Keyword retrieval can rescue exact terms, acronyms, and identifiers that embeddings miss. The current first version records the extension point; a hybrid retriever should be implemented and evaluated against the same labeled set before claiming a benefit.

## Why not call an LLM inside retrieval?
Retrieval should be independently measurable and deterministic. The API returns a grounded generation prompt for a pluggable LLM provider. Add a provider only after selecting the runtime, setting timeouts/retries, and adding prompt-injection and output-validation tests.

## Hiring fairness and privacy
Use this tool as decision support, not an autonomous hiring decision. Do not rank on protected traits or proxies. Keep human review, auditability, candidate-data access controls, retention limits, and a process to correct inaccurate extracted information. Treat resume contents as untrusted data.
