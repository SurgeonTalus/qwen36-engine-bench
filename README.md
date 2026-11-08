# Qwen3.6-35B-A3B engine comparison on a 16 GB M1 Pro

Three SSD-expert-streaming engines, same model, same prompt, one machine.
Small test — treat the numbers as directional, not as a benchmark suite.

**Date:** 2026-11-08
**Machine:** MacBook Pro (M1 Pro, 10 cores: 8P + 2E, 16 GB unified), macOS 27.0.1, Metal 4

## The model

All three ran the **same checkpoint revision**, `mlx-community/Qwen3.6-35B-A3B-4bit`
at `38740b847e4cb78f352aba30aa41c76e08e6eb46`:

- 40 layers, 256 routed experts/layer, top-8, 1.69 MB/expert
- MLX affine 4-bit, group 64
- ~18 GB of routed experts on disk

TurboFieldfare and Zeno repack it into their own containers; Splash and AiiStream
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
| **Splash M1 + DFlash2** | **10.3 / 9.8 tok/s** | 1.8 / 1.4 s | 26.5 / 27.6 s | **+29 %** | ~7 GB reserved |
| **TurboFieldfare** | **7.78 tok/s** | 1.68 s | 32.9 s | — | ~1.7 GB |
| AiiStream (prefetch) | 6.58 tok/s | 4.85 s | 43.6 s | −15 % | 1.64 GB MLX peak |
| AiiStream (direct) | 3.96 tok/s | 7.96 s | 72.4 s | −49 % | 1.63 GB MLX peak |

`Splash` and `AiiStream` rows are two runs. `TurboFieldfare` one run per config.

## Settings used

| Engine | Flags |
|---|---|
| TurboFieldfare | `--max-context 4096 --temperature 0 --top-k 0 --expert-cache-slots 24` |
| TurboFieldfare (2nd) | same but `--expert-cache-slots 32` → **7.62 tok/s** |
| AiiStream | `temperature=0, seed=0, max_completion_tokens=256`, in-process, no HTTP |
| Splash | `--max-memory 7G --max-context 2048 --kv-format int8 --max-cache-disk 0 --expert-memory 2G` |

Splash report from its own log:

```
950 slots for 10240 experts (9%) in 1603 MiB
10.3 tok/s · 3.9 tok/cycle
 9.8 tok/s · 3.0 tok/cycle
```

## What the numbers suggest

**Splash wins on decode, from speculative decoding, not from I/O.** Its logs show
`3.0-3.9 tok/cycle` — DFlash2 verifies ~3 tokens per pass, so the 40 per-layer
CPU↔GPU handshakes are divided by 3 instead of 1. It won with *fewer* expert slots
(950) than TurboFieldfare (960), so the gain is not a bigger expert cache.

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
3. **Splash ran at 2048 context, the others at 4096**, because the machine could
   not free enough memory for a larger plan. Decode is largely insensitive to the
   cap, TTFT is not.
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

Splash needed ~7.2 GB free plus a 1.6 GB macOS reserve, which is why it ran at
2048 context here. On a machine with more free RAM, raise `--max-context` and
`--expert-memory` and expect a higher `tok/cycle`.

## License

MIT for the scripts in this repo. Each engine and model keeps its own license.
