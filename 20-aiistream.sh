#!/usr/bin/env bash
# AiiStream — same prompt, greedy, 256 tokens. Runs the repo's own paired A/B:
# `direct` (on-demand) vs `prefetch` (look-ahead), in-process, no HTTP.
set -uo pipefail
cd "$HOME/aiistream-q3.6"
PROMPT='Explain in detail how a hash table handles collisions, with examples.'
TOKENS=256

echo "### AiiStream"
echo "swap before: $(sysctl -n vm.swapusage)"
.venv/bin/python - "$PROMPT" "$TOKENS" <<'PY'
import gc, sys, time
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "server"))
import mlx.core as mx
from aiistream_server.runtime_engine import RuntimeEngine
from aiistream_server.safety import SafetyState

prompt, tokens = sys.argv[1], int(sys.argv[2])
res = {}
for mode in ("direct", "prefetch"):
    engine = RuntimeEngine(repo_root=ROOT, safety=SafetyState(), expert_read_path=mode)
    try:
        prepared = engine.prepare([{"role": "user", "content": prompt}])
        mx.reset_peak_memory()
        r = engine.generate_prepared(prepared, temperature=0.0, seed=0,
                                     max_completion_tokens=tokens)
        decode_s = r.elapsed_s - (r.ttft_s or 0.0)
        res[mode] = dict(ids=r.token_ids, n=r.completion_tokens,
                         decode=(r.completion_tokens - 1) / decode_s if decode_s > 0 else None,
                         ttft=r.ttft_s, total=r.elapsed_s,
                         peak_gb=mx.get_peak_memory() / 1e9)
    finally:
        engine.close(); del engine; gc.collect(); mx.clear_cache()

for m, v in res.items():
    print(f"{m:9s} decode {v['decode']:.2f} tok/s | ttft {v['ttft']:.2f}s | "
          f"total {v['total']:.1f}s | {v['n']} tok | mlx peak {v['peak_gb']:.2f} GB")
print("identical tokens:", res["direct"]["ids"] == res["prefetch"]["ids"])
PY
echo "swap after: $(sysctl -n vm.swapusage)"
