#!/usr/bin/env bash
# Hybrid-retrieval experiments on the server (paper "Knowing When to Adapt"). Resumable: each finished step leaves a
# marker in state/, so re-running the script continues where it stopped. Nothing here touches the registry: own
# PostgreSQL container (hybrid-exp-pg, plain postgres:16), own port bound to 127.0.0.1 only.
#
#   ./run_all.sh                 # everything
#   ./run_all.sh preflight       # checks only
#   STEPS="bm25 fts" ./run_all.sh
# Options (environment): PYTHON, PG_PORT (18508), THREADS (all cores), LLM_MODEL (qwen2.5:7b),
#   OLLAMA_HOST_URL (http://127.0.0.1:11434), WITH_FIQA_LLM=1 to also run the LLM controllers on FiQA (648 queries).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p state logs results data models
PG_PORT="${PG_PORT:-18508}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 4)}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
DSN="postgresql://postgres:postgres@127.0.0.1:${PG_PORT}/fts"
export THREADS LLM_MODEL OLLAMA_HOST_URL FASTEMBED_CACHE="$HERE/models" HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$HERE/code"
STEPS="${STEPS:-${1:-preflight venv data postgres bm25 fts llm package}}"
[ "${1:-}" = "preflight" ] && STEPS="preflight"

if [ -z "${PYTHON:-}" ]; then
  for p in python3.12 python3.11; do command -v "$p" >/dev/null && PYTHON="$p" && break; done
fi
PY="$HERE/.venv/bin/python"
log() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }
done_mark() { touch "state/$1.done"; }
is_done() { [ -f "state/$1.done" ]; }
want() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

# ------------------------------------------------------------------ preflight
if want preflight; then
  log "preflight -> results/preflight.txt"
  {
    echo "date: $(date -Is)"; echo "host: $(uname -a)"; echo "cpus: $(nproc 2>/dev/null)"
    free -g 2>/dev/null | head -2; df -h "$HERE" | tail -1
    echo "python: ${PYTHON:-NONE} $(${PYTHON:-false} --version 2>&1 || true)"
    echo "docker: $(docker version --format '{{.Server.Version}}' 2>&1 || true)"
    echo "port ${PG_PORT}: $( (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${PG_PORT} " && echo BUSY || echo free)"
    echo "ollama: $(curl -s --max-time 5 "$OLLAMA_HOST_URL/api/tags" | tr -d '\n' | head -c 600 || echo UNREACHABLE)"
    echo "huggingface: $(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://huggingface.co || echo FAIL)"
    echo "beir: $(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/ || echo FAIL)"
  } | tee results/preflight.txt
  fail=0
  [ -n "${PYTHON:-}" ] || { echo "ERREUR : python3.12 ou python3.11 introuvable (définir PYTHON=...)"; fail=1; }
  for c in curl unzip tar; do command -v "$c" >/dev/null || { echo "ERREUR : $c absent (sudo apt install $c)"; fail=1; }; done
  docker version >/dev/null 2>&1 || { echo "ERREUR : Docker inaccessible (sudo, ou ajouter l'utilisateur au groupe docker)"; fail=1; }
  if grep -q "port ${PG_PORT}: BUSY" results/preflight.txt && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx hybrid-exp-pg; then
    echo "ERREUR : port ${PG_PORT} occupé (définir PG_PORT=...)"; fail=1
  fi
  avail=$(df -Pk "$HERE" | awk 'NR==2{print int($4/1048576)}')
  [ "$avail" -ge 15 ] || { echo "ERREUR : ${avail} Go libres, 15 Go nécessaires"; fail=1; }
  if ! grep -q "\"name\":\"${LLM_MODEL}\"" results/preflight.txt; then
    echo "ATTENTION : modèle ${LLM_MODEL} absent d'Ollama ($OLLAMA_HOST_URL) -> étape llm impossible (ollama pull ${LLM_MODEL}, ou STEPS sans llm)"
  fi
  [ "$fail" = 0 ] || exit 1
  echo "preflight OK"
fi

# ------------------------------------------------------------------ venv
if want venv && ! is_done venv; then
  log "Python environment ($PYTHON)"
  [ -d .venv ] || "$PYTHON" -m venv .venv
  .venv/bin/pip install -q --no-cache-dir -r requirements.txt 2>&1 | tail -3
  "$PY" -c "import fastembed, sklearn, psycopg, rank_bm25, snowballstemmer, numpy; print('imports OK')"
  done_mark venv
fi

# ------------------------------------------------------------------ data
if want data && ! is_done data; then
  log "BEIR datasets"
  for d in scifact nfcorpus trec-covid fiqa; do
    if [ ! -f "data/$d/corpus.jsonl" ]; then
      tmp=$(mktemp -d)
      curl -sSfL -o "$tmp/$d.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip"
      (cd "$tmp" && unzip -q "$d.zip") && mv "$tmp/$d" "data/$d" && rm -rf "$tmp"
    fi
    echo "$d: $(wc -l < "data/$d/corpus.jsonl") documents, $(($(wc -l < "data/$d/qrels/test.tsv") - 1)) test qrels"
  done
  done_mark data
fi

# ------------------------------------------------------------------ postgres
pg_up() {
  if ! docker ps -a --format '{{.Names}}' | grep -qx hybrid-exp-pg; then
    docker run -d --name hybrid-exp-pg -p "127.0.0.1:${PG_PORT}:5432" -e POSTGRES_USER=postgres \
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=fts postgres:16 >/dev/null
  else
    docker start hybrid-exp-pg >/dev/null
  fi
  until docker exec hybrid-exp-pg pg_isready -U postgres -d fts >/dev/null 2>&1; do sleep 2; done
}
if want postgres && ! is_done postgres; then
  log "PostgreSQL 16 (hybrid-exp-pg, 127.0.0.1:${PG_PORT})"
  pg_up
  docker exec hybrid-exp-pg psql -U postgres -d fts -tAc "select version()"
  done_mark postgres
fi

# ------------------------------------------------------------------ BM25 rankings / matrices
if want bm25; then
  for d in hemophilia-clin trec-covid fiqa; do
    is_done "bm25-$d" && continue
    log "BM25 + dense: $d"
    "$PY" code/utility_matrix_beir.py --data "data/$d" --out "results/$d" --threads "$THREADS" 2>&1 | tee "logs/bm25-$d.log" | tail -2
    done_mark "bm25-$d"
  done
fi

# ------------------------------------------------------------------ database full-text search
if want fts; then
  pg_up
  for d in hemophilia-dev scifact nfcorpus hemophilia-clin trec-covid fiqa; do
    is_done "fts-$d" && continue
    log "FTS (PostgreSQL) + dense: $d"
    "$PY" code/utility_matrix_fts.py --data "data/$d" --out "results/$d-fts" --dsn "$DSN" --threads "$THREADS" 2>&1 \
      | tee "logs/fts-$d.log" | tail -2
    done_mark "fts-$d"
  done
fi

# ------------------------------------------------------------------ LLM controllers B1/B2
if want llm; then
  curl -s --max-time 5 "$OLLAMA_HOST_URL/api/tags" | grep -q "\"name\":\"${LLM_MODEL}\"" \
    || { echo "ERREUR : modèle ${LLM_MODEL} indisponible sur ${OLLAMA_HOST_URL}"; exit 1; }
  if ! is_done llm-smoke; then
    log "LLM smoke test (3 queries)"
    "$PY" code/llm_features.py --data data/hemophilia-dev --out results/llm-smoke --limit 3 2>&1 | tail -1
    done_mark llm-smoke
  fi
  LLM_SETS="hemophilia-dev hemophilia-clin scifact nfcorpus trec-covid"
  [ "${WITH_FIQA_LLM:-0}" = 1 ] && LLM_SETS="$LLM_SETS fiqa"
  for d in $LLM_SETS; do
    is_done "llm-$d" && continue
    log "LLM controllers B1/B2 ($LLM_MODEL): $d"
    "$PY" code/llm_features.py --data "data/$d" --out "results/$d-llm" 2>&1 | tee "logs/llm-$d.log" | tail -1
    done_mark "llm-$d"
  done
fi

# ------------------------------------------------------------------ package
if want package; then
  out="hybrid_results_$(date +%Y%m%d_%H%M).tar.gz"
  log "package -> $out"
  tar czf "$out" results logs state code/prompts.py
  ls -lh "$out"
  echo "Envoyer ce fichier : $HERE/$out"
fi
log "fin des étapes demandées : $STEPS"
