# Stage-5 end-to-end smoke

Driver: `vsag-code` console script (package `vsag_code`) running headless
(`--once`) against the same SIFT-1M dataset used by the Stage-1 spike.
This run exercises the full agent loop: schema-validated tool registry,
LLM provider client, permission gate, index registry, and event trace
writer.

## Gate criteria

The smoke passes if the agent, given a single natural-language goal,
chains the registered tools end-to-end and obtains a recall@10 within
5 percentage points of the spike baseline (~0.95) on full SIFT-1M.

## Environment

- Pod: `vsag-dev-5dd59d5667-w4nn2` (HouseBrain `k8s/apps/vsag-dev/`,
  ns `vsag-dev` on station microk8s)
- Node: `station` (AMD Ryzen 7 5700G, 8c16t, AVX2; no AVX512)
- Code: `feature/vsag-code-tui-agent` @ `192d8cdd`
- Dataset: `/data/datasets/sift-128-euclidean.hdf5`
  (1M × 128 train, 10K test, 100-NN ground truth, euclidean)
- Provider: `deepseek` / model `deepseek-chat`
- Invocation:

  ```bash
  PYTHONPATH=agent:$PYTHONPATH python3 -m vsag_code \
      --provider deepseek --no-color \
      --once 'Inspect /data/datasets/sift-128-euclidean.hdf5, build an
              hgraph index on the full training set, run search with
              topk=10 ef_search=100 over all 10000 test queries, and
              report recall@10, p95 latency, and QPS.' \
      --out /workspace/logs/smoke-stage5.json
  ```

## Trace summary

Four LLM steps, four tool dispatches, no permission prompts (all tools
in this goal are `read` or `mutate`, none `destructive`):

| Step | Tool          | Key arguments                                               | Outcome                            |
|-----:|---------------|-------------------------------------------------------------|------------------------------------|
|    1 | `dataset_info`| `path=/data/datasets/sift-128-euclidean.hdf5`               | dim=128, train=1M, test=10K, l2    |
|    2 | `index_build` | `algorithm=hgraph` (preset: sq8, max_degree=26, efc=100)    | handle=`hgraph-idx-1`, build=962.67s |
|    3 | `index_search`| `handle=hgraph-idx-1, topk=10, ef_search=100`               | recall@10=0.9946, p95=1.637ms      |
|    4 | (final)       | natural-language summary                                    | see "Reported numbers" below       |

Total wall: ~16 min (dominated by `index_build`).
LLM round-trip: 1.49s + 1.78s + 2.32s + 2.44s = 8.0s.

## Measured numbers (from `index_search` tool result)

| Metric          | Value     |
|-----------------|----------:|
| num_queries     | 10,000    |
| recall@10       | **0.9946**|
| mean latency    | 1.392 ms  |
| p50 latency     | 1.418 ms  |
| p95 latency     | 1.637 ms  |
| p99 latency     | 1.729 ms  |
| QPS             | 712.2     |
| build_seconds   | 962.67    |

Recall (0.9946) clears the gate by a wide margin against the ~0.95
baseline, and is materially higher than the Stage-1 spike's 0.9565 on
the same hardware/dataset. The improvement is attributable to the
Stage-2 preset actually nesting hgraph build params under the
`"hgraph"` key (sq8 / max_degree=26 / efc=100), which the spike's flat
`index_param` shape did not propagate to pyvsag.

## LLM-reported numbers vs. measured

The agent's final natural-language paragraph reads:

> recall@10 = 0.9946, p95 latency = 1.64 ms, and QPS = 7,122

Recall and p95 match the tool result exactly. The reported QPS of
**7,122 is wrong by 10×** — the actual measured QPS is **712.2**. The
model appears to have misplaced the decimal when summarising. The
trace JSON preserves the authoritative tool result, so downstream
consumers should read numbers from `trace[].result` rather than the
final assistant message. A future hardening step is to teach the
prompt (or a verifier tool) to echo numbers verbatim from the last
`index_search` result.

## Artifacts

- `agent/spike/RESULTS.md` — Stage-1 spike (baseline)
- Pod-side: `/workspace/logs/smoke-stage5.log`,
  `/workspace/logs/smoke-stage5.json`
- Local copies: `/tmp/smoke-stage5.log`, `/tmp/smoke-stage5.json`
  (147-line JSON trace + 38-line ANSI-stripped console log)

## Dual-provider gate (mirrors Stage-1 spike)

The same goal was re-run with `--provider copilot` (model
`gpt-4o`, GitHub Copilot Chat). Same pod, same dataset, fresh
index registry. Three tool dispatches plus the closing summary:

| Metric          | DeepSeek (`deepseek-chat`) | Copilot (`gpt-4o`) |
|-----------------|--------------------------:|-------------------:|
| build_seconds   | 962.67                    | 964.24             |
| num_queries     | 10,000                    | 10,000             |
| recall@10       | 0.9946                    | 0.9945             |
| mean latency    | 1.392 ms                  | 1.383 ms           |
| p95 latency     | 1.637 ms                  | 1.618 ms           |
| p99 latency     | 1.729 ms                  | 1.695 ms           |
| QPS             | 712.2                     | 717.1              |
| LLM steps       | 4 (1.49 + 1.78 + 2.32 + 2.44 s) | 4 (2.33 + 2.35 + 3.09 + 4.25 s) |
| QPS in summary  | 7,122 (10× wrong)         | 717.1 (correct)    |

Copilot's `gpt-4o` reproduced the run within noise on every metric
and reported the QPS verbatim. The DeepSeek 10× misreport in the
final paragraph appears to be model-specific. The trace JSON is
authoritative either way.

Copilot artifacts:

- `/workspace/logs/smoke-stage5-copilot.log` (pod)
- `/workspace/logs/smoke-stage5-copilot.json` (pod)
- `/tmp/smoke-stage5-copilot.{log,json}` (local)

## Status

Stage 5 gate: **PASS** under both providers. Recall is within
tolerance for DeepSeek and Copilot, the agent loop runs unattended
end-to-end against real `pyvsag` on real SIFT-1M, and the trace
writer captures every tool call with arguments and results.
