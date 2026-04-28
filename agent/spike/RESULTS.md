# Stage-1 spike: results

Driver: [`spike.py`](spike.py) calling tools in [`tools.py`](tools.py).
Tools register three functions: `dataset_info`, `index_build`, `index_search`.

## Gate criteria (proposal §5.2)

The spike passes if **at least one** LLM provider can chain the three
tools end-to-end from a natural-language goal and obtain a recall@10
within 5 percentage points of a known-good baseline on SIFT-1M
(hgraph defaults: ~0.95 recall@10).

## Environment

- Pod: `vsag-dev` (HouseBrain `k8s/apps/vsag-dev/`)
- Node: `station` (microk8s, AMD Ryzen 7 5700G, 8c16t, **no AVX512**;
  vsag falls back to AVX2 at runtime)
- Build: `make dev` against
  `feature/vsag-code-tui-agent` @ `cba20024`
- Dataset: `/data/datasets/sift-128-euclidean.hdf5`
  (1M × 128 train, 10K test, 100-NN ground truth, euclidean,
  md5 `52ee452f4fb106133dd2d420bc1bb1aa`)
- Python deps: `numpy 2.2.6`, `h5py 3.16.0` (installed via direct
  wheel fetch; pip 26.1 resolver is unusable on this pod).

## Tools-only smoke (no LLM)

| Run            | Algorithm | Train rows | Build (s) | Queries | recall@10 | Mean lat (ms) | p99 (ms) | QPS  |
|----------------|-----------|-----------:|----------:|--------:|----------:|--------------:|---------:|-----:|
| 100k subset    | hgraph    |    100,000 |    106.0  |   1,000 | 0.1636 *  |        0.969  |    1.128 | 1019 |
| Full SIFT-1M   | hgraph    |  1,000,000 |   1352.4  |   1,000 | **0.9565**|        1.133  |    1.382 |  873 |

\* The 100k recall is an artifact: the dataset's GT neighbor ids span
the full 1M corpus; only ~16% are below 100k, which exactly matches
the observed recall. The tools are correct; the appropriate smoke
config is the full 1M build.

The full-1M run clears the gate by a comfortable margin (recall
0.9565 vs. ~0.95 expected for hgraph defaults).

## Provider runs

### DeepSeek (`deepseek-chat`, OpenAI-compatible)

Goal:
> Inspect /data/datasets/sift-128-euclidean.hdf5, build a hgraph index
> on it, then report recall@10 and latency over 1000 queries.

_TBD: paste step trace + final answer + verdict_

### GitHub Copilot (`gpt-4o`, device-flow)

_TBD_

## Verdict

_TBD: pass / fail per provider; gate decision; observed regressions._

## Notes for stage 2

- `tools.TOOLS` is already a list of `ToolSpec` objects with
  OpenAI-style JSON schema; stage 2 can lift this directly.
- Index registry is process-local; stage 2 needs a serialization
  story (vsag's `serialize`/`deserialize` API + a TTL cache).
- Latency stats currently use a 1-sample p99 estimator; stage 2
  should switch to `numpy.percentile` once the dependency is
  unconditional.
