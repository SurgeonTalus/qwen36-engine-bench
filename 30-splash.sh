#!/usr/bin/env bash
# Splash M1 — same prompt, greedy, 256 tokens, via the prepared 35B assembly
# with the incoai DFlash2 draft (speculative decoding ON, which is the point).
set -uo pipefail
CUR="$HOME/Library/Application Support/Splash-M1/current"
ASSEMBLY="$HOME/Library/Application Support/Splash/models/.resolved/758afbbc2edce7b50dcaeaa5b75033727a5c31828283140e401713c69bfcd666"
PROMPT='Explain in detail how a hash table handles collisions, with examples.'

echo "### Splash M1"
echo "wired limit before: $(sysctl -n iogpu.wired_limit_mb)"

echo "$PROMPT" > /tmp/splash-messages.json
python3 - <<'PY'
import json
prompt = 'Explain in detail how a hash table handles collisions, with examples.'
json.dump([{"role": "user", "content": prompt}], open('/tmp/splash-messages.json', 'w'))
print("messages written")
PY

echo "swap before: $(sysctl -n vm.swapusage)"
echo
echo "--- start server (background) ---"
PYTHONPATH="$CUR" "$CUR/python/bin/python3" -P -m server.server "$ASSEMBLY" \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --binary "$CUR/engine/splash" \
  --host 127.0.0.1 --port 8099 \
  --max-memory 10G \
  --kv-format int8 \
  > /tmp/splash-server.log 2>&1 &
SPID=$!
echo "server pid $SPID"

for i in $(seq 1 90); do
  if curl -s -m 2 http://127.0.0.1:8099/v1/models >/dev/null 2>&1; then
    echo "ready after ~$((i*2))s"; break
  fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; tail -20 /tmp/splash-server.log; exit 1; fi
  sleep 2
done

curl -s -m 5 http://127.0.0.1:8099/v1/models; echo
echo "--- request (greedy, 256 tokens) ---"
/usr/bin/time -l curl -sS -m 900 http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit","messages":[{"role":"user","content":"Explain in detail how a hash table handles collisions, with examples."}],"temperature":0,"max_tokens":256,"stream":false}' \
  -o /tmp/splash-response.json 2>&1 | tail -8

python3 - <<'PY'
import json
try:
    d = json.load(open('/tmp/splash-response.json'))
except Exception as e:
    print("parse error:", e); raise SystemExit
u = d.get('usage', {})
print("usage:", json.dumps(u))
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '')
print("chars:", len(msg), "| head:", msg[:120].replace("\n", " "))
if u.get('completion_tokens'):
    pass
PY

echo "--- server log tail ---"
tail -15 /tmp/splash-server.log
echo "swap after: $(sysctl -n vm.swapusage)"
kill $SPID 2>/dev/null
echo "server stopped"
