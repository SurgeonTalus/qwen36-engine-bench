#!/usr/bin/env bash
# TurboFieldfare — greedy, no system prompt, 256 tokens, phase counters on.
set -uo pipefail
CLI="$HOME/turbo-fieldfare-qwen36-precache/.build/release/TurboFieldfareCLI"
MODEL="$HOME/.local/share/turbofieldfare/qwen36.gturbo"
PROMPT='Explain in detail how a hash table handles collisions, with examples.'

echo "### TurboFieldfare"
echo "swap before: $(sysctl -n vm.swapusage)"
TURBO_FIELDFARE_PHASES=1 /usr/bin/time -l "$CLI" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max-new 256 \
  --max-context 4096 \
  --temperature 0 \
  --top-k 0 \
  --expert-cache-slots 24 \
  2>&1 | tail -30
echo "swap after: $(sysctl -n vm.swapusage)"
