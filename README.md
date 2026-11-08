# Qwen3.6-35B-A3B engine comparison on a 16 GB M1 Pro

Three SSD-expert-streaming engines, same model, same prompt, one machine.
Small test — treat the numbers as directional, not as a benchmark suite.

**Date:** 2026-11-08
**Machine:** MacBook Pro (M1 Pro, 10 cores: 8P + 2E, 16 GB unified), macOS 27.0.1, Metal 4

## The engines

| Engine | Project | Version used |
|---|---|---|
| TurboFieldfare | <https://github.com/drumih/turbo-fieldfare> | local fork with disk prefix cache, `--max-context 65536` support |
| Splash (M1 build) | <https://github.com/incoai/splash> · M1/M2 fork: <https://github.com/paperniuk/splash> | `splash-m1` 1.4.0-m1 |
| AiiStream Q3.6 | <https://github.com/apirathaiya/aiistream-q3.6> | `main` at time of test, `mlx==0.32.2`, `mlx-lm==0.31.3` |

Model source: <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit>

## The model

All three ran the **same checkpoint revision**, `mlx-community/Qwen3.6-35B-A3B-4bit`
at `38740b847e4cb78f352aba30aa41c76e08e6eb46`:

- 40 layers, 256 routed experts/layer, top-8, 1.69 MB/expert
- MLX affine 4-bit, group 64
- ~18 GB of routed experts on disk

TurboFieldfare repacks it into its own `.gturbo` container; Splash and AiiStream
read the MLX safetensors directly. Same weights, same routing.

## Test

```
prompt:       "Explain in detail how a hash table handles collisions, with examples."
messages:     user only (no system prompt)
sampling:     greedy, temperature 0
tokens:       256 generated
decode metric: tokens / (total - TTFT), prefill excluded
```

No system prompt and no prefix cache on purpose: this measures raw decode, not
the prompt-reuse that all three engines can do.

## Results

| Engine | Decode | TTFT | Total (256 tok) | vs TurboFieldfare | Peak RAM |
|---|---:|---:|---:|---:|---:|
| **Splash M1 + DFlash2, 8G / 4K** | **12.3 / 13.2 tok/s** | 1.3 s | 20.7 / 20.2 s | **+64 %** | ~8 GB reserved |
| **Splash M1 + DFlash2, 7G / 2K** | 10.3 / 9.8 tok/s | 1.8 / 1.4 s | 26.5 / 27.6 s | +29 % | ~7 GB reserved |
| **TurboFieldfare** (24 slots) | **7.78 tok/s** | 1.68 s | 32.9 s | — | ~1.7 GB |
| AiiStream (prefetch) | 6.58 tok/s | 4.85 s | 43.6 s | −15 % | 1.64 GB MLX peak |
| AiiStream (direct) | 3.96 tok/s | 7.96 s | 72.4 s | −49 % | 1.63 GB MLX peak |

Splash scales with the memory it gets: 7G/2K gave 950 expert slots and 9.8–10.3 tok/s,
8G/4K gave 1557 slots and 12.3–13.2 tok/s. It did not fit at 10G / 4G expert memory on
this machine (`not enough free memory to start`, 1587 MiB reclaimable).

`Splash`, `AiiStream` and `TurboFieldfare` rows are two runs where shown.

### Splash cannot run without the draft

Tried removing `draft/` from the prepared assembly to free ~1 GB and isolate the
effect of speculative decoding. The engine refuses:

```
MODEL_DIRECTORY must hold the model's target/ and draft/ directories
```

So DFlash2 is not optional in this build, and the contribution of `tok/cycle`
can't be isolated by simply switching it off.

## Settings used

| Engine | Flags |
|---|---|
| TurboFieldfare | `--max-context 4096 --temperature 0 --top-k 0 --expert-cache-slots 24` |
| TurboFieldfare (2nd) | same but `--expert-cache-slots 32` → **7.62 tok/s** |
| AiiStream | `temperature=0, seed=0, max_completion_tokens=256`, in-process, no HTTP |
| Splash (7G) | `--max-memory 7G --max-context 2048 --kv-format int8 --max-cache-disk 0 --expert-memory 2G` |
| Splash (8G) | `--max-memory 8G --max-context 4096 --kv-format int8 --max-cache-disk 0 --expert-memory 3G` |

Splash reports from its own logs:

```
7G / 2K:  950 slots for 10240 experts (9%) in 1603 MiB
          10.3 tok/s · 3.9 tok/cycle   |   9.8 tok/s · 3.0 tok/cycle
8G / 4K:  1557 slots for 10240 experts (15%) in 2627 MiB
          12.3 tok/s · 3.0 tok/cycle   |  13.2 tok/s · 2.8 tok/cycle
```

## What the numbers suggest

**Splash wins on decode, and the margin depends on how much memory it gets.** At
7 GB it gave 9.8–10.3 tok/s (+29 % over TurboFieldfare); at 8 GB with a 4K context
it gave 12.3–13.2 tok/s (**+64 %**). Its logs show `2.8–3.9 tok/cycle` — DFlash2
verifies ~3 tokens per pass, so the 40 per-layer CPU↔GPU handshakes are divided by
3 instead of 1. The 8G run won with 1557 expert slots against TurboFieldfare's 960,
but the 7G run won with 950 — fewer — so the gain is not simply a bigger expert cache.

**AiiStream's prefetch works but starts from a lower baseline.** It is +66 % over
its own `direct` mode (3.96 → 6.58), which matches its published +72 %. But that
`direct` mode is plain on-demand streaming without overlap — not TurboFieldfare.

**TurboFieldfare's decode is not I/O-bound on this machine.** From its phase counters:

```
24 slots:  I/O await 10.9 s (33%) | GPU waits 21.0 s (64%) | 7.78 tok/s
32 slots:  I/O await 12.6 s (38%) | GPU waits 19.7 s (60%) | 7.62 tok/s
```

So the ceiling for perfect I/O hiding is only ~1.5×, and a bigger expert cache
measured *worse* — explicit slots displacing the OS page cache. The large
remaining cost is GPU dispatch/latency: 82 ms/token against ~1.6 GB of weights
read, where bandwidth alone would predict ~8 ms.

## Caveats — read these before quoting anything

1. **Single machine, single session.** M1 Pro 16 GB, load average 5–19, 1.7–9 GB
   of swap in use at various points. Not a clean bench.
2. **Runs were sequential, not interleaved.** Paired A/B with alternating order is
   the correct method for I/O-sensitive engines; this was not that. Treat the
   magnitudes as indicative.
3. **Splash's context varied with available memory**: 2048 at 7 GB, 4096 at 8 GB, and
   it would not start at 10 GB / 4 GB expert memory. Decode is largely insensitive
   to the cap, TTFT is not. The machine was also carrying 1.7–1.9 GB of swap.
4. **TurboFieldfare was measured at 24 and 32 slots; the app defaults to 16.**
   At 16 it would be ~3–5 % slower than the 7.78 shown.
5. **Splash is not bit-identical run to run.** Its own log says so
   (`greedy and seeded outputs can differ from run to run`). AiiStream and
   TurboFieldfare produced identical tokens across their two modes.
6. **Different quantisation of the *same* checkpoint** — all MLX affine 4-bit
   group 64, but each engine has its own repack/kernels. Not bit-comparable across
   engines.

## Why this is even worth publishing

The published numbers from each project come from different hardware and compare
against different baselines. Splash's headline is against M5 with the model
resident; AiiStream's +72 % is against its own on-demand mode; TurboFieldfare's
are from an M5 Pro. This is one machine, one model, one prompt, three engines —
which none of them publish.

## Reproducing

Scripts in this repo are the ones actually used, kept as-is including the
workarounds. They assume:

- `~/.local/share/turbofieldfare/qwen36.gturbo` — TurboFieldfare model
- `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit` — MLX checkpoint
- Splash M1 installed (`splash-m1`, release 1.4.0-m1) plus a prepared assembly

### Get the pieces

```sh
# 1. The checkpoint (~20 GB, SHA-256 verified per shard)
hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46

# 2. TurboFieldfare — streams the same checkpoint from the Hub and repacks it
git clone https://github.com/drumih/turbo-fieldfare
cd turbo-fieldfare
swift build -c release
swift run -c release TurboFieldfareRepack --model qwen36 --output scratch/qwen36.gturbo

# 3. Splash M1 (M1/M2 need the fork — upstream requires M3+)
curl -fsSL https://github.com/paperniuk/splash/releases/download/1.4.0-m1/install-m1.sh | sh

# 4. AiiStream
git clone https://github.com/apirathaiya/aiistream-q3.6
cd aiistream-q3.6 && python3 -m venv .venv && source .venv/bin/activate
pip install "mlx==0.32.2" "mlx-lm==0.31.3" huggingface_hub
scripts/download_model.sh    # ~20 GB, verified
```

Splash needs a DFlash2 draft for the target family. It fetches
`incoai/Qwen3.6-35B-A3B-DFlash2` (~2 GB) when preparing a local model, or take it
from the Hub yourself. Note that `z-lab/Qwen3.6-35B-A3B-DFlash` is a *different*
draft format and Splash rejects it.

Splash needed ~7.2 GB free plus a 1.6 GB macOS reserve, which is why it ran at
2048 context here. On a machine with more free RAM, raise `--max-context` and
`--expert-memory` and expect a higher `tok/cycle`.

## Related work

- TurboFieldfare performance notes: <https://github.com/drumih/turbo-fieldfare/blob/main/docs/QWEN36_PERFORMANCE.md>
- Splash documentation: <https://github.com/incoai/splash/tree/main/docs>
- AiiStream benchmark and identity scripts: <https://github.com/apirathaiya/aiistream-q3.6/tree/main/scripts>

A closely related project not tested here: <https://github.com/Niko1221/Strata>
(GGUF + Metal, MoE expert streaming). It does not currently run Qwen3.6 on the
Mac engine, so it was out of scope.



## License

MIT for the scripts in this repo. Each engine and model keeps its own license.
