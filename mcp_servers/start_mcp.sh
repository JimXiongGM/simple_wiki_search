#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"

EMBEDDING_SESSION="simple-wiki-embedding"
WIKIPEDIA_SESSION="simple-wiki-server"

kill_session() {
  local name="$1"
  if tmux has-session -t "=${name}" 2>/dev/null; then
    tmux kill-session -t "=${name}"
  fi
}

start_embedding() {
  kill_session "${EMBEDDING_SESSION}"
  tmux new-session -d -s "${EMBEDDING_SESSION}" bash -lc "
    cd '${REPO_ROOT}'
    mkdir -p '${LOG_DIR}'
    python -u -m serving.embedding_router \
      --model-name Qwen/Qwen3-Embedding-0.6B \
      --start-port 17000 > >(tee -a '${LOG_DIR}/embedding_api.log') 2>&1
  "
  echo "embedding api: tmux attach -t ${EMBEDDING_SESSION}"
}

start_wikipedia() {
  kill_session "${WIKIPEDIA_SESSION}"
  tmux new-session -d -s "${WIKIPEDIA_SESSION}" bash -lc "
    cd '${REPO_ROOT}'
    mkdir -p '${LOG_DIR}'
    python -u -m mcp_servers.wikipedia_offline_server --host 127.0.0.1 --port 11536 \
      --embedding_api_url http://localhost:17000 \
      > >(tee -a '${LOG_DIR}/wikipedia_offline_server.log') 2>&1
  "
  echo "wikipedia server: tmux attach -t ${WIKIPEDIA_SESSION}"
}

# start_embedding
start_wikipedia
