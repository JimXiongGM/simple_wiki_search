# Deploy

End-to-end setup: download indexes → embedding → MCP tools → QA data → actor LLM → run agent (or reuse published outputs).

## 0. Prerequisites and environment

Launch scripts assume **Linux** (Bash, `tmux`, `fork`). For the full index, plan on ~32–64 GiB RAM and ~100 GiB free disk. A local embedding server or actor needs an NVIDIA GPU and CUDA/SGLang; remote APIs work instead, and the Wikipedia tool server itself does not need a GPU.

Install the system tools and common Python dependencies:

```bash
sudo apt-get update
sudo apt-get install -y tmux zstd

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -c 'import sglang; print(sglang.__version__)'
```

Indexes, benchmarks, and published outputs:

- [Google Drive](https://drive.google.com/drive/folders/1WO-7eswLCVvsNJM2pmEwYyoImyoN2hXr?usp=sharing)
- [Baidu Pan](https://pan.baidu.com/s/517dqRhQ4-5UgqjTeGPDK6Q)

## 1. Download and extract Wikipedia indexes

Two archives are required:

| Archive | Contents |
|---------|----------|
| `enwiki-20260601-faiss-hnsw-Qwen3-Embedding-0.6B.tar.zst` | FAISS HNSW dense index |
| `enwiki-20260601-tantivy-chunk-1536.tar.zst` | Tantivy keyword index + chunk text store |

Extract (needs `zstd`):

```bash
tar -I zstd -xf enwiki-20260601-faiss-hnsw-Qwen3-Embedding-0.6B.tar.zst
tar -I zstd -xf enwiki-20260601-tantivy-chunk-1536.tar.zst
```

After extraction the layout should look like:

```text
data/database/wikipedia-index/enwiki-20260601/
  faiss/hnsw-Qwen3-Embedding-0.6B-chunk-1536-fp16/
    index.faiss
    ids.pkl
  tantivy/chunk-1536/
    chunks/
    titles/
```

These match the defaults in `settings.py` (`INDEX_PATHS`).

## 2. Configure an embedding endpoint

Dense / RRF search needs an OpenAI-compatible `/v1/embeddings` endpoint for `Qwen/Qwen3-Embedding-0.6B` (dim=1024). **At least one** provider must be available.

Detection order is **local first, then online API**. See [`tools/wikipedia_offline/embedding_detect.py`](tools/wikipedia_offline/embedding_detect.py).

**Option A: local (recommended when you have a GPU)**

```bash
python -m serving.embedding_router \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --start-port 17000
```

Default local URL: `http://127.0.0.1:17000` (see `DEFAULT_EMBEDDING_API_URL` in `settings.py`).

**Option B: online API fallback**

If local probe fails, detection falls through to the online candidate documented in `embedding_detect.py` (set the env var that file expects).

## 3. Start the MCP / tool server and smoke-test

With indexes in place and an embedding endpoint reachable:

```bash
bash mcp_servers/start_mcp.sh
```

This starts the Wikipedia offline tool server on `http://127.0.0.1:11536` (tmux session `simple-wiki-server`). Logs: `logs/wikipedia_offline_server.log`.

Smoke tests:

```bash
python -m mcp_servers.wikipedia_offline_client --search "capital of China" --top_k 5
python -m mcp_servers.wikipedia_offline_client --search "Albert Einstein" --method keywords --top_k 5
python -m mcp_servers.wikipedia_offline_client --open_url "https://en.wikipedia.org/wiki/Albert_Einstein"
```

If search / open_url return hits and page text, the tool stack is up.

## 4. Download QA benchmarks

From the same Drive folder, download `qa_benchmarks_6.tar.zst` and extract:

```bash
tar -I zstd -xf qa_benchmarks_6.tar.zst
```

Expected layout:

```text
data/dataset/
  full_test/{2WikiMultiHopQA,HotpotQA,MuSiQue,FRAMES,PopQA,Bamboogle}/...
  random_300/{2WikiMultiHopQA,HotpotQA,MuSiQue,FRAMES,PopQA,Bamboogle}.json
```

## 5. Start an actor LLM (and optional LLM judge)

**Local open-weight actor** (SGLang behind a small router on port `19000`):

```bash
python -m serving.llm_router \
  --model-name Qwen/Qwen3.5-9B \
  --start-port 19000
```

Other local models used in the baseline table work the same way (e.g. `Qwen/Qwen3.5-4B`, `google/gemma-4-12B-it`, `mistralai/Ministral-3-8B-Instruct-2512-BF16`).

**Remote / closed-source actor.** The baseline scripts accept `PROVIDER`, `BASE_URL`, and `API_KEY` as `KEY=VALUE` arguments; direct `run_qa_agent` calls use `--provider`, `--base_url`, and `--api_key`. Prefer a provider-specific environment variable so the secret is not placed on the command line:

```bash
# OpenAI: provider selects https://api.openai.com/v1 and reads OPENAI_API_KEY.
export OPENAI_API_KEY='...'
bash scripts/run_agent_baseline_random_300.sh \
  MODEL_NAME=gpt-5.4-2026-03-05 PROVIDER=openai
```

`PROVIDER` can be `openai`, `openrouter`, or `deepseek`; each reads its corresponding API-key environment variable. For another OpenAI-compatible service, set `BASE_URL` (with or without `/v1`) and `OPENAI_API_KEY`, or explicitly provide `API_KEY`. `BASE_URL` takes priority over `PROVIDER`.

**LLM judge (optional).** Baseline run scripts pass `--enable_llm_judge`. When enabled, set up the judge API as in [`evaluation/llm_judge.py`](evaluation/llm_judge.py) (uses `provider="openai"`, so `OPENAI_API_KEY` must be set). Without judge, you still get deterministic text F1; the `gpt` column will be empty / unavailable.

## 6. Run the agent

Full test:

```bash
bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=Qwen/Qwen3.5-9B
# OpenAI actor (after exporting OPENAI_API_KEY as above)
bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=gpt-5.4-2026-03-05 PROVIDER=openai
```

Cheaper 300-example subset:

```bash
bash scripts/run_agent_baseline_random_300.sh MODEL_NAME=Qwen/Qwen3.5-9B
```

Outputs land under `output_full_test/` or `output_sample_random_300/` as `{dataset}/{model_dir}/*.json`.

## 7. Or download published results

From Drive, download `output_full_test.tar.zst` and `output_sample_random_300.tar.zst`, then:

```bash
tar -I zstd -xf output_full_test.tar.zst
tar -I zstd -xf output_sample_random_300.tar.zst
python scripts/print_qa_benchmark_md_table.py
```

This reprints the Markdown score tables (same form as in [README.md](README.md)).
