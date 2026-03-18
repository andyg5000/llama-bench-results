# Qwen3.5-35B-A3B Q8_0 — Autoresearch Results

**Hardware:** NVIDIA GB10 (Project DIGITS), 128GB unified memory, ARM Cortex-X925 20-core
**Model:** Qwen3.5-35B-A3B-Q8_0.gguf (35GB)
**Date:** 2026-03-18
**Method:** Autoresearch (LLM-driven experiment loop)

## Results Summary

| # | Experiment | Ctx | Cache K/V | Batch/UB | Par | Coding | Agentic | Speed | Context | Gen t/s | PP t/s | Score | Kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | Baseline f16 262k | 262k | f16/f16 | 2048/512 | 1 | C | B- | B+ | B | 50.0 | 1768 | 0.6360 | seed |
| 002 | Baseline f16 65k | 65k | f16/f16 | 1024/256 | 1 | C | B- | B+ | A+ | 50.1 | 1343 | 0.6553 | yes |
| 003 | **q8_0 cache 65k** | 65k | q8_0/q8_0 | 1024/256 | 1 | A- | A | A | A+ | 50.1 | 1341 | **0.7460** | yes |
| 004 | q8_0 131k ctx | 131k | q8_0/q8_0 | 1024/256 | 1 | A- | A | A | B | 50.0 | 1340 | 0.7155 | no |
| 005 | q8_0 131k (retry) | 131k | q8_0/q8_0 | 1024/256 | 1 | A- | A | A | B | 50.3 | 1341 | 0.7165 | no |
| 006 | **q8_0 batch 2048** | 65k | q8_0/q8_0 | 2048/512 | 1 | A- | A | A | A+ | 50.1 | 1760 | **0.7572** | yes |
| 007 | q8_0 131k batch 2048 | 131k | q8_0/q8_0 | 2048/512 | 1 | A- | A | A | B | 50.0 | 1744 | 0.7264 | no |
| 008 | q8_0 batch 4096 | 65k | q8_0/q8_0 | 4096/1024 | 1 | B | A | A | A+ | 50.4 | 2084 | 0.7366 | no |
| 009 | q8_0 batch 4096 (retry) | 65k | q8_0/q8_0 | 4096/1024 | 1 | B | A | A | A+ | 50.0 | 2099 | 0.7361 | no |

**Converged after 9 experiments. Best composite score: 0.7572**

## Key Findings

### q8_0 KV cache is a major accuracy win

Switching from f16 to q8_0 cache jumped coding from C to A- and agentic from B- to A, boosting composite score from 0.6553 to 0.7460. This was the single largest improvement found.

### 65k context outperforms 131k

At 131k context, the `ctx_xlarge` needle-in-haystack test failed consistently, dropping context grade from A+ to B. The 65k context passed all needle tests. This held across both batch size configurations.

### Batch 2048/512 is the sweet spot

- **1024/256:** 1341 t/s PP — lower throughput
- **2048/512:** 1760 t/s PP — good balance, coding accuracy maintained at A-
- **4096/1024:** 2084-2099 t/s PP — fastest prompt processing, but coding accuracy dropped to B

The batch 4096 config trades ~13% coding accuracy for ~19% faster prompt processing. Not worth it for the composite score.

### Generation speed is GPU compute-bound at ~50 t/s

Generation speed is constant at ~50 t/s regardless of cache type, context size, batch size, or any other parameter. The GB10 is the bottleneck.

## Recommended Config

```
--ctx-size 65536
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 2048
--ubatch-size 512
--parallel 1
--n-gpu-layers 100
--flash-attn on
```

**Result:** 65k context, ~50 t/s generation, ~1760 t/s prompt processing, A-/A/A/A+ grades across coding/agentic/speed/context profiles.

## Scoring Methodology

Composite score weights:
- **Speed (30%):** accuracy (60%) + gen t/s normalized to 100 (40%)
- **Coding (30%):** accuracy (60%) + response time factor (40%)
- **Context (20%):** needle accuracy (60%) + PP t/s normalized to 3000 (40%)
- **Agentic (20%):** accuracy (50%) + concurrent accuracy (50%)

## Experiment Methodology

Autoresearch loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

1. Research LLM proposes next experiment config based on results history
2. Starts a fresh llama-server with the test configuration
3. Evaluates across 4 profiles: coding (8 tasks), agentic (8 tasks + concurrency), speed (12 tasks), context (4 tests + gen speed)
4. Computes composite score and keeps/discards based on improvement
5. Converges after 3 consecutive experiments with no improvement

All generation TPS values are server-reported (`predicted_per_second`), not wall-clock estimates.
