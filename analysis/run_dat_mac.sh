#!/usr/bin/env bash
# DAT baseline (dat_features.py) on the Mac (Apple M1, 8 GB), memory-safe overnight run. Resumable (llm_cache.jsonl per corpus).
# Stop at any time:  touch ~/Downloads/RAG/experiments/llm_mac/STOP   (or: pkill -f run_llm_mac.sh; pkill -f dat_features.py)
set -uo pipefail
L="$(cd "$(dirname "$0")" && pwd)"
E=/Users/hb/Downloads/RAG/experiments
C=$E/server_bundle/code
H=$E/hybrid/data
R=$E/server_results/20261117_2021/results
LOG=$L/run_dat_mac.log
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
exec >>"$LOG" 2>&1
rm -f "$L/STOP"
echo "== $(date '+%F %T') start"
caffeinate -i -w $$ &
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null; then
  nohup /opt/homebrew/opt/ollama/bin/ollama serve >>"$L/ollama.log" 2>&1 &
  for i in $(seq 1 60); do curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
fi
curl -s http://127.0.0.1:11434/api/version; echo
# watchdog: macOS memory pressure level (1 normal, 2 warning, 4 critical); stop only after 2 consecutive critical
# readings (60 s); logs the level and swap every 10 min
( crit=0; n=0
  while sleep 30; do
    [ -f "$L/STOP" ] && exit 0
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
    n=$((n + 1)); [ $((n % 20)) = 0 ] && echo "   [$(date '+%T')] memory pressure level $lvl, $(sysctl -n vm.swapusage | awk '{print "swap used", $7}')"
    if [ "$lvl" -ge 4 ]; then crit=$((crit + 1)); else crit=0; fi
    if [ "$crit" -ge 4 ]; then
      echo "== $(date '+%F %T') WATCHDOG: critical memory pressure for 2 min -> stopping"
      touch "$L/STOP"; pkill -f dat_features.py; exit 0
    fi
  done ) &
WD=$!
status=complete
run() {  # name data rankings
  [ -f "$L/STOP" ] && { status=INCOMPLETE; return; }
  [ -f "$L/$1-dat/DONE" ] && return
  mkdir -p "$L/$1-dat"
  echo "== $(date '+%F %T') $1"
  if PYTHONPATH=$C nice -n 10 "$L/.venv/bin/python" $C/dat_features.py --data "$2" --out "$L/$1-dat" --rankings "$3" \
       --host http://127.0.0.1:11434; then
    touch "$L/$1-dat/DONE"
  else
    status=INCOMPLETE; touch "$L/STOP"
  fi
}
run hemophilia-dev  $H/hemophilia-dev  $L/rankings/hemophilia-dev/rankings.jsonl.gz
run hemophilia-clin $H/hemophilia-clin $R/hemophilia-clin/rankings.jsonl.gz
run scifact         $H/scifact         $L/rankings/scifact/rankings.jsonl.gz
run nfcorpus        $H/nfcorpus        $L/rankings/nfcorpus/rankings.jsonl.gz
run trec-covid      $H/trec-covid      $R/trec-covid/rankings.jsonl.gz
kill $WD 2>/dev/null
echo "== $(date '+%F %T') $status"
pkill -f "ollama serve"
