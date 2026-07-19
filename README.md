# SimpleWikiSearch: A Clean Offline Wikipedia Environment for Agentic Search

## Motivation

Agentic-search results are hard to compare when the environment is underspecified. Many papers report final QA scores but omit details that materially affect behavior: which Wikipedia database is used, how pages are cleaned, the chunk size, retrieval backend, tool schema, observation format, observation truncation, whether multiple tool calls are allowed in one step, stopping rules, and evaluation normalization.

This repository provides one end-to-end baseline instead of another loosely described setup. It starts from a full English Wikipedia dump, cleans and chunks the corpus, builds Tantivy keyword and FAISS HNSW vector indexes, exposes a minimal tool interface (`search`, `open_url`, `submit_answer`), and runs a simple Wikipedia-only QA agent. The goal is not to claim a new reasoning method; the goal is a clean, inspectable reference environment that future agentic-search work can cite, run, and modify.

Fixed 100-word passage corpora are no longer a good default for modern tool-using LLMs. This repo uses a full-dump pipeline and evaluates modern models on standard QA benchmarks, with both full-test results and a cheaper random-300 subset for API models.

## Corpus and Indexes

| Item | Setting |
|------|---------|
| Snapshot | `enwiki-20260601` |
| Articles | **7,189,602** |
| Chunks | **10,837,506** |
| Chunking | section-aware, target **1536** tokens (`Qwen3-Embedding-0.6B` tokenizer, 20% tolerance) |
| Keyword index | Tantivy (`tantivy/chunk-1536`), same `chunk_id` set as FAISS |
| Dense index | FAISS HNSW fp16, `Qwen3-Embedding-0.6B`, dim=1024 |
| Default retrieval | RRF over keyword + dense |

Dense search returns `chunk_id` values only (e.g. `736_0`). Chunk text and titles are resolved by looking up those ids in Tantivy. The keyword index matches the FAISS id set one-to-one (10,837,506 chunks), so every vector hit has a corresponding offline text payload.

Index layout under `data/database/wikipedia-index/enwiki-20260601/`:

```text
tantivy/chunk-1536/{chunks,titles}                # keyword + text store (FAISS-aligned)
faiss/hnsw-Qwen3-Embedding-0.6B-chunk-1536-fp16/  # index.faiss + ids.pkl
```

Reproduction instructions are in [deploy.md](deploy.md).

## Baseline Results

Scores are percentages. `f1` is deterministic text F1; `gpt` is LLM-judge accuracy. The `sample_random_300` subset uses 300 examples per dataset (Bamboogle keeps all 125).

| Dataset | Full-test size |
|---------|---------------:|
| 2Wiki | 12576 |
| HotpotQA | 7405 |
| MuSiQue | 2417 |
| FRAMES | 824 |
| PopQA | 14267 |
| Bamboogle | 125 |

full_test

| Method | 2Wiki |  | HotpotQA |  | MuSiQue |  | FRAMES |  | PopQA |  | Bamboogle |  |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  | f1 | gpt | f1 | gpt | f1 | gpt | f1 | gpt | f1 | gpt | f1 | gpt |
| Ministral-3-8B-Instruct-2512-BF16 | 40.24 | 64.14 | 32.62 | 58.03 | 15.13 | 24.90 | 22.78 | 33.13 | 42.57 | 57.52 | 42.04 | 64.00 |
| Qwen3.5-4B | 76.44 | 86.28 | 53.95 | 70.93 | 29.07 | 35.37 | 42.35 | 54.85 | 53.93 | 65.57 | 66.24 | 73.60 |
| Qwen3.5-9B | 75.59 | 87.76 | 55.18 | 73.36 | 32.58 | 40.50 | 48.12 | 60.80 | 56.57 | 65.84 | 69.30 | 73.60 |
| gemma-4-12B-it | 66.33 | 81.18 | 41.50 | 60.24 | 18.03 | 23.00 | 22.00 | 27.91 | 43.00 | 67.03 | 57.13 | 62.40 |

sample_random_300

| Method | 2Wiki |  | HotpotQA |  | MuSiQue |  | FRAMES |  | PopQA |  |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  | f1 | gpt | f1 | gpt | f1 | gpt | f1 | gpt | f1 | gpt |
| Ministral-3-8B-Instruct-2512-BF16 | 37.23 | 61.67 | 35.84 | 63.33 | 15.59 | 28.00 | 23.78 | 33.00 | 44.52 | 57.33 |
| Qwen3.5-4B | 77.73 | 86.33 | 56.99 | 71.33 | 28.89 | 33.67 | 42.81 | 55.67 | 53.32 | 65.00 |
| Qwen3.5-9B | 82.29 | 89.67 | 59.46 | 73.00 | 33.09 | 43.67 | 46.20 | 57.67 | 57.95 | 64.00 |
| gemma-4-12B-it | 66.12 | 80.00 | 39.59 | 56.00 | 19.12 | 26.33 | 23.55 | 28.33 | 40.38 | 63.67 |
| deepseek-v4-pro | 67.30 | 90.67 | 60.79 | 79.00 | 31.44 | 44.67 | 67.04 | 85.67 | 53.05 | 73.33 |
| gpt-5.4-2026-03-05 | 67.88 | 90.33 | 63.37 | 87.67 | 39.94 | 58.33 | 65.63 | 84.33 | 56.60 | 73.00 |

## Citation

TBC