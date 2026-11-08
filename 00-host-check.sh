#!/usr/bin/env bash
# Paired token-throughput test: TurboFieldfare vs AiiStream vs Splash M1
# Same prompt, greedy, no system prompt, 256 generated tokens.
set -uo pipefail

PROMPT='Explain in detail how a hash table handles collisions, with examples.'
TOKENS=256

echo "=================== HOST STATE ==================="
date
uptime
sysctl -n vm.swapusage
echo
echo "--- competing processes ---"
pgrep -fl 'xcodebuild|swift-frontend|mlx_lm|llama-server|mlx_lm.server' | head -5 || echo "(none)"
echo
echo "--- memory pressure ---"
memory_pressure 2>/dev/null | grep -i "free percentage"
echo
