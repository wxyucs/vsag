# VSAG-Code: A Conversational TUI Agent for VSAG

> **Status**: Proposal / Pre-MVP
> **Branch**: `feature/vsag-code-tui-agent`
> **Author**: drafted with AI assistance, to be picked up in a long-running dev environment
> **Audience**: VSAG maintainers / contributors who will implement the MVP

This document is **self-contained**. It captures:

1. The original discussion (research → proposal → reframing)
2. The final design (a conversational TUI agent in the spirit of OpenCode / Claude Code / Aider)
3. All context needed to resume implementation in a fresh environment without re-doing the analysis

---

## Table of Contents

- [0. TL;DR](#0-tldr)
- [1. Background and Original Conversation](#1-background-and-original-conversation)
  - [1.1 First question: should VSAG have a CLI?](#11-first-question-should-vsag-have-a-cli)
  - [1.2 Reframe: a TUI agent, not a CLI toolset](#12-reframe-a-tui-agent-not-a-cli-toolset)
- [2. VSAG Capability Inventory](#2-vsag-capability-inventory)
- [3. Final Design: VSAG-Code](#3-final-design-vsag-code)
  - [3.1 Form factor](#31-form-factor)
  - [3.2 Mapping from OpenCode primitives to VSAG primitives](#32-mapping-from-opencode-primitives-to-vsag-primitives)
  - [3.3 Example interactions](#33-example-interactions)
  - [3.4 Architecture](#34-architecture)
  - [3.5 Tool registry (the soul of the agent)](#35-tool-registry-the-soul-of-the-agent)
  - [3.6 What is unique vs. a generic coding agent](#36-what-is-unique-vs-a-generic-coding-agent)
  - [3.7 Risks and mitigations](#37-risks-and-mitigations)
- [4. Implementation Plan](#4-implementation-plan)
  - [4.1 Recommended order of work](#41-recommended-order-of-work)
  - [4.2 MVP scope (2–3 weeks)](#42-mvp-scope-23-weeks)
  - [4.3 Repository layout](#43-repository-layout)
  - [4.4 Tech stack and rationale](#44-tech-stack-and-rationale)
- [5. Resume Kit (read this first when picking up)](#5-resume-kit-read-this-first-when-picking-up)
- [6. Open Questions](#6-open-questions)
- [Appendix A. Concrete VSAG facts referenced by the agent](#appendix-a-concrete-vsag-facts-referenced-by-the-agent)
- [Appendix B. Verbatim conversation log](#appendix-b-verbatim-conversation-log)

---

## 0. TL;DR

Build **`vsag-code`**: a conversational TUI agent (think *OpenCode for vector indexes*) that lets users operate the VSAG library through natural-language dialogue. The LLM does not generate code — it calls a curated set of **domain-specific tools** that wrap VSAG's existing C++/Python surface (`pyvsag` + the existing `tools/eval`, `tools/analyze_index`, `tools/check_compatibility` binaries).

Why it works for VSAG specifically:

- VSAG operations are **structured** (JSON params), **reversible** (clone / save), and produce **quantitative feedback** (recall, QPS, p99, memory). This is the ideal substrate for an LLM agent loop — much friendlier than freeform code editing.
- VSAG already ships strong prior art: a working eval CLI, an index analyzer, ~37 examples, and a parameter recipe table in `benchs/README.md`. The agent's job is to **orchestrate** these, not to reimplement them.
- The agent can self-close the loop (target recall ≥ X, p99 ≤ Y) and run long benchmarks in the background — uses the TUI form factor in a way a normal CLI cannot.

MVP target: **2–3 weeks**, Python + [Textual](https://textual.textualize.io/) + OpenAI-compatible LLM SDK + 8 core tools.

---

## 1. Background and Original Conversation

### 1.1 First question: should VSAG have a CLI?

The user originally asked:

> "VSAG/ is our project … CLI is a hot concept right now. Analyze our current tooling/debugging/testing capabilities and tell me whether we can build a VSAG CLI tool — possibly wrapping all current capabilities, even supporting LLM API config and natural-language operation of the library."

After full repository exploration (see [§2](#2-vsag-capability-inventory)), the verdict was:

- **Yes, a CLI is feasible.** VSAG already has 80% of a CLI hidden inside `tools/eval/main.cpp` and the bindings.
- A clean CLI tree was sketched, e.g.:

  ```
  vsag build / search / add / remove / merge / clone / export-model
  vsag bench / analyze / stats / estimate-memory / check-compat
  vsag tune / dataset download / dataset info
  vsag dev test|fmt|lint
  vsag ai "<natural-language goal>"
  ```

The user accepted the analysis but **rejected the form factor**.

### 1.2 Reframe: a TUI agent, not a CLI toolset

The user clarified:

> "Not bad, but my understanding of CLI is different. I want a TUI tool — conversational, like OpenCode."

This changed the target product entirely:

| Was proposed (rejected) | What is actually wanted |
|---|---|
| A bag of subcommands (`vsag build …`) | A persistent REPL / TUI session |
| LLM is one optional subcommand | LLM is the primary driver; tools are how it acts |
| User composes commands | User states goals; agent plans + executes |
| One-shot invocations | Multi-turn conversation, background tasks, todos |

The rest of this document describes that revised design.

---

## 2. VSAG Capability Inventory

This section is a digest of the codebase exploration. It is the **input knowledge** the agent's tool layer wraps.

### 2.1 Library surface (`include/vsag/`)

- **Index algorithms** (`include/vsag/constants.h`): `hnsw`, `fresh_hnsw`, `diskann`, `hgraph`, `ivf`, `pyramid`, `brute_force`, `sparse`, `sindi`, `gno_imi`. Enum `IndexType { HNSW, DISKANN, HGRAPH, IVF, PYRAMID, BRUTEFORCE, SPARSE, SINDI }`.
- **Data types**: `float32`, `float16`, `bfloat16`, `int8`, `sparse` (CSR).
- **Metrics**: `l2`, `ip`, `cosine`.
- **Quantization codecs**: `fp32`, `fp16`, `bf16`, `sq4_uniform`, `sq8_uniform`, `pq`, `rabitq`.
- **Public headers a CLI/agent calls into**:

  | Header | Role |
  |---|---|
  | `factory.h` | `Factory::CreateIndex(name, json)` → main entry |
  | `engine.h` | Lifecycle, allocator, thread pool |
  | `index.h` | The `Index` class (Build/Add/Remove/Search/Serialize/Tune/…) |
  | `dataset.h` | Builder for input batches |
  | `search_request.h`, `search_param.h`, `filter.h`, `bitset.h` | Search inputs |
  | `binaryset.h`, `readerset.h` | Serialization containers |
  | `resource.h`, `allocator.h`, `thread_pool.h` | Customization |
  | `attribute.h`, `index_features.h`, `index_detail_info.h` | Feature flags + introspection |
  | `iterator_context.h` | Iterator-mode search state |
  | `vsag_c_api.h`, `vsag_ext.h` | C ABI / extension API |
  | `errors.h`, `expected.hpp` | `tl::expected`-based error handling |
  | `constants.h` | All algorithm/dtype/metric/quantization name strings |

- **`Index` methods relevant as tools**: `Build`, `Add`, `Remove`, `UpdateId`, `UpdateVector`, `KnnSearch`, `RangeSearch`, `SearchWithRequest`, `Serialize`, `Deserialize`, `Merge`, `Clone`, `ExportModel`, `Train`, `Tune`, `Pretrain`, `Feedback`, `CalcDistanceById`, `GetStats`, `AnalyzeIndexBySearch`, `EstimateMemory`, `GetIndexDetailInfos`, `GetNumElements`, `GetMemoryUsage`, `CheckIdExist`, `GetMinAndMaxId`.

### 2.2 Existing tools (`tools/`, gated by `ENABLE_TOOLS=ON`)

| Binary | Purpose | Args (abridged) |
|---|---|---|
| `tools/eval/eval_performance` | Benchmark / eval — single-test argparse mode and YAML batch mode | `--datapath`, `--type {build,search}`, `--index_name`, `--create_params` (JSON), `--index_path`, `--search_params` (JSON), `--search_mode {knn,range,knn_filter,range_filter}`, `--topk`, `--range`, `--search-query-count`, plus disable-flags for individual metrics |
| `tools/analyze_index/analyze_index` | Static + dynamic index diagnostics for HGraph/IVF/Pyramid | `--index_path`, `--build_parameter`, `--query_path`, `--search_parameter`, `--topk` |
| `tools/check_compatibility/check_compatibility` | CI backward-compat check | `<tag>_<algo>` (expects `/tmp/<tag>_<algo>.index` etc.) |

`tools/eval` already has subdirs for cases (`build_eval_case`, `search_eval_case`, `build_search_eval_case`), exporters (stdout / file / appendfile / influxdb; table / json / markdown / line_protocol), and monitors (duration, latency, memory_peak, recall, http_server). Reports include QPS, TPS, recall (avg + percentiles), latency (avg + p50/p95/p99), peak memory.

### 2.3 Benchmarks (`benchs/`)

Pure YAML configs consumed by `eval_performance` (no separate binary).

- `benchs/datasets/sift-90.yml` — same SIFT dataset across HGRAPH/HNSW/IVF/DISKANN at 90% recall.
- `benchs/indexes/hgraph-{90,95,99}.yml` — HGRAPH across DEEP1B / GIST / SIFT / GloVe-100 / NYTimes / COHERE / OpenAI.
- `benchs/eval_template.yaml` — template.
- `benchs/README.md` — **the canonical recipe table**: parameter recipes per (algorithm × dataset × recall). This is gold for RAG.

### 2.4 Tests (`tests/`)

- **Catch2** (C++): two binaries — `unittests` (per-module: `simd_test`, `vsag_test`, `algorithm_test`, `factory_test`, `attr_test`, `datacell_test`, `quantizer_test`, `storage_test`, `io_test`, `utils_test`, `impl_test`) and `functests` (~25 `tests/test_*.cpp` files).
- **mockimpl/tests_mockimpl** for mock-impl tests.
- **Fixtures** (`tests/fixtures/`): `recall_checker`, `test_dataset`, `test_dataset_pool`, `test_logger`, `test_reader`, `vector_generator`, `ground_truth`, `temp_dir`, `allocator`/`core`/`data`/`framework` helpers — directly reusable for agent integration tests.
- **Python tests** (`tests/python/`, pytest): `test_bruteforce.py`, `test_dataset.py`, `test_hgraph.py`, `test_hnsw.py`, `test_index_operations.py`, `test_ivf.py`, plus `conftest.py`, `run_test.py`, `test_runner.sh`.
- **Sanitizers**: `make asan` / `make test_asan`, `make tsan` / `make test_tsan`. **Coverage**: `make cov`.

### 2.5 Examples (`examples/`)

- **C++ (`examples/cpp/`, ~37 files)**:
  - Algorithm starters (1xx): `101_hnsw`, `102_diskann`, `103_hgraph`, `104_fresh_hnsw`, `105_brute_force`, `106_ivf`, `107_pyramid`, `108_gno_imi`, `109_sindi`.
  - Resources (2xx): `201_custom_allocator`, `202_custom_logger`, `203_custom_thread_pool`.
  - Features (3xx): filter, range_search, remove, enhance_graph, update, calc_distance_by_id, check_features, estimate_memory, clone, export_model, train, odescent, search_allocator, hgraph_search_allocator, hgraph_merge, int8_hgraph, get_detail_data, tune, get_memory_usage, extra_info, fp16_hgraph.
  - Persistence (4xx): `401_persistent_kv`, `402_persistent_streaming`.
  - Quantization (5xx): `501_quantization_transform`.
- **Python (`examples/python/`)**: `101_hnsw`, `102_diskann`, `103_hgraph`, `105_brute_force`, `106_ivf`, `109_sindi`, plus older `example_diskann`, `example_hnsw`. `todo_examples/` for not-yet-ported features.
- **TypeScript (`examples/typescript/`)**: `101_index_hnsw.ts` only.

### 2.6 Bindings

- **Python** (`python/`, `python_bindings/`): pybind11 module `_pyvsag`, repackaged as `pyvsag` (`python/pyvsag/__init__.py`, `python/setup.py`). Class `Index(name: str, parameters: str)`; methods `build` (dense + CSR sparse overloads), `knn_search` (dense + sparse), `range_search`, `save`, `load`, `add`, `remove`, `get_num_elements`, `get_memory_usage`, `check_id_exist`, `get_min_max_id`, `cal_distance_by_id`, plus logging-level bindings. Build: `make pyvsag PY_VERSION=3.10` or `make pyvsag-all`. See `scripts/python/local_build_wheel.sh`, `scripts/python/prepare_python_build.sh`.
- **Node / TypeScript** (`node_bindings/`, `typescript/`): N-API via cmake-js → `build/Release/vsag_node.node`. Package `vsag` v0.1.0. TS API (`typescript/src/index.ts`): `Index` with `build`, `add`, `remove`, `knnSearch`, `rangeSearch`, `save`, `load`, `getNumElements`, `getMemoryUsage`, `checkIdExist`, `getMinMaxId`, `calDistanceById`, plus `setLoggerOff/Info/Debug`.
- **C ABI**: `include/vsag/vsag_c_api.h`.

### 2.7 Build system & scripts

- **CMake ≥ 3.18**, GCC ≥ 9.4 or Clang ≥ 13, C++17.
- Top-level `Makefile` targets: `help`, `debug`, `dev`, `test`, `asan`, `test_asan`, `tsan`, `test_tsan`, `clean`, `fmt`, `cov`, `lint`, `fix-lint`, `test_parallel`, `test_asan_parallel`, `test_tsan_parallel`, `release`, `run-dist-tests`, `dist-pre-cxx11-abi`, `dist-cxx11-abi`, `dist-libcxx`, `pyvsag` (with `PY_VERSION=3.10`), `pyvsag-all`, `clean-release`, `install`.
- CMake options: `ENABLE_TESTS`, `ENABLE_EXAMPLES`, `ENABLE_TOOLS`, `ENABLE_PYBINDS`, `ENABLE_MOCKIMPL`, `ENABLE_INTEL_MKL`, `USE_SYSTEM_OPENBLAS`, `ENABLE_LIBAIO`, `ENABLE_ASAN`, `ENABLE_TSAN`, `ENABLE_COVERAGE`, `ENABLE_CCACHE`, `ENABLE_LIBCXX`, `ENABLE_CXX11_ABI`, `ENABLE_WERROR`. `make dev` enables everything.
- **Scripts** (`scripts/`):
  - `format/` — `format-cpp.sh`, `check_format.sh` (clang-format-15)
  - `linters/` — `run-clang-tidy-15.sh`, `run-clang-tidy.py`
  - `coverage/` — `check_cov.sh`, `collect_cpp_coverage.sh`, `gcov_for_clang.sh`
  - `python/` — `local_build_wheel.sh`, `prepare_python_build.sh`
  - `release/` — `check-cpp-abi.sh`, `dist.sh`
  - `testing/` — `test_parallel_bg.sh`, `test_parallel_by_name.sh`
  - `deps/` — `install_deps_ubuntu.sh`, `install_deps_centos.sh`
  - `perf_reports/dingding.py` — DingTalk perf reporting
  - `csv_extract/csv_extract.py`
  - Top-level: `download_annbench_datasets.sh`, `check_compatibility.sh`, `check_environment.sh`, `change_mtime.sh`

### 2.8 Configuration format

All index/algorithm parameters are JSON strings. Two common shapes:

**Legacy / flat** (HNSW, DiskANN, BruteForce, etc.):

```json
{
  "dim": 128,
  "dtype": "float32",
  "metric_type": "l2",
  "hnsw": { "max_degree": 16, "ef_construction": 100 }
}
```

**Newer wrapped** (HGraph, IVF, Pyramid):

```json
{
  "dim": 128,
  "dtype": "float32",
  "metric_type": "l2",
  "index_param": {
    "base_quantization_type": "sq8_uniform",
    "max_degree": 32,
    "ef_construction": 400
  }
}
```

**Search params**:
- HNSW: `{"hnsw": {"ef_search": 100}}`
- HGraph: `{"hgraph": {"ef_search": 60}}`
- IVF: `{"ivf": {"scan_buckets_count": 13}}`
- DiskANN: `{"diskann": {"ef_search": 120, "beam_search": 4, "io_limit": 60, "use_reorder": true}}`

**YAML batch wrapper** (eval mode) embeds these JSON strings as `create_params` / `search_params` per case — see `benchs/eval_template.yaml`.

### 2.9 Contribution rules (must-follow when modifying core repo, not the agent)

From `AGENTS.md`: Google C++ style, 4-space indent, `.cpp` extension, 100-char lines, **clang-format-15 / clang-tidy-15 exact versions**, prefer `uint64_t` over `size_t`, DCO sign-off, Conventional Commits, PR labels `kind/*` + `version/*`.

The agent itself (Python) is exempt from C++ rules but should still use Conventional Commits + DCO sign-off.

---

## 3. Final Design: VSAG-Code

### 3.1 Form factor

A standalone TUI binary, launched by:

```bash
vsag-code              # opens conversational TUI
vsag-code --resume     # resume last session
vsag-code --headless   # one-shot, no TUI (for CI / scripts)
```

Inside the TUI: persistent dialogue, tool-call cards, todo panel, background-task panel, optional table/chart panes. **Not** a subcommand tree.

### 3.2 Mapping from OpenCode primitives to VSAG primitives

| OpenCode | VSAG-Code |
|---|---|
| User chats with LLM | Same |
| LLM calls tools to read/write code | LLM calls tools to operate VSAG indexes |
| `Read`/`Write`/`Edit` files | `index.build`, `index.add`, `index.remove`, `index.save`/`load`, `dataset.load` |
| `Bash` to run commands | Run `eval_performance`, `analyze_index`, `make test`, `python -m pytest` |
| `Grep`/`Glob` to search code | `index.search` (KNN/Range), `dataset.filter`, `examples.find` |
| Todo-list tool for plans | Multi-step experiment plans (build → tune → bench → compare) |
| Files in working tree as context | Indexes, datasets, param JSONs, benchmark results as context |

### 3.3 Example interactions

**Scenario A — goal-driven build:**

```
> 帮我用 SIFT-1M 跑一下 HGraph，目标 95% 召回

● [tool: dataset.list]
  → 找到 ~/data/sift-128-euclidean.hdf5

● 根据 benchs/README.md 的 95% 召回推荐：
    base_quantization_type: sq8_uniform
    max_degree: 32, ef_construction: 400
  [tool: index.build]  预估 ~3 分钟

  ▸ 执行？ (y/n/edit)
> y

  ✓ build 完成，4096MB，2m41s
  [tool: bench.run]
  ✓ recall@10 = 0.947, QPS = 8240, p99 = 1.3ms

● 略低于 0.95，要把 ef_search 从 60 调到 100 再试吗？
> 调到 80 试试

  [tool: bench.run] ef_search=80
  ✓ recall@10 = 0.953, QPS = 7100, p99 = 1.6ms ✓
```

**Scenario B — diagnosis:**

```
> 我有个 idx.bin，最近召回掉了，帮我看看

● [tool: index.stats] idx.bin
  → 1.2M vectors, hgraph, sq8_uniform, 3.8GB
● [tool: index.analyze] 1000 queries
  → connect_components: 23 ⚠ (正常应 = 1)
  → quantization_bias_ratio: 0.31 ⚠ (>0.2 偏高)
  → duplicate_rate: 0.08

● 两个问题：
  1. 图有 23 个连通分量 — 检查 build log 是否跳过了向量
  2. sq8 量化偏差偏高 — 试试 export_model 后改 fp16 量化？
```

**Scenario C — autonomous tuning loop:**

```
> 把这个索引调到 recall>0.95 且 p99<2ms

● 计划：
  1. baseline bench
  2. 二分 ef_search 找到 recall≥0.95 的下界
  3. 在该下界附近微调 max_degree
  4. 验收并报告
  [confirm? y/n]
> y

  [10 轮 tune+bench，TUI 显示 Pareto 表格实时更新]
  ✓ 收敛：ef_search=72, max_degree=36 → recall=0.954, p99=1.7ms
```

### 3.4 Architecture

```
┌────────────────────────────────────────────────────────┐
│              TUI Layer (Textual)                       │
│  对话流 / 工具调用卡片 / diff / 表格 / 进度条 / Pareto 图 │
└────────────────────────────────────────────────────────┘
                        │
┌────────────────────────────────────────────────────────┐
│              Agent Loop                                │
│  多轮对话 / tool calling / 任务规划 / 上下文压缩       │
│  Provider: OpenAI / Anthropic / DeepSeek / Ollama 等   │
└────────────────────────────────────────────────────────┘
                        │
┌────────────────────────────────────────────────────────┐
│              Tool Registry  (核心)                     │
│  index.* | dataset.* | bench.* | params.* | dev.* |    │
│  docs.*  | session.* | bash (受限)                     │
└────────────────────────────────────────────────────────┘
                        │
┌────────────────────────────────────────────────────────┐
│              VSAG Library                              │
│  pyvsag  +  subprocess(eval_performance/analyze_index) │
└────────────────────────────────────────────────────────┘
```

### 3.5 Tool registry (the soul of the agent)

Tool design principles:

- **Two-tier**: a small set of high-level tools that LLM prefers (`bench.run`, `index.build_for_recall_target`) which delegate to atomic tools (`index.build`, `index.search`). High-level tools cut token usage and round-trips.
- **Permission tiers**: `read` (auto), `mutate` (confirm by default), `destructive` (explicit `--yes` or interactive confirm).
- **Structured errors**: `{code, message, suggestion}` so the LLM can self-recover.
- **Idempotency where possible**: `index.build` writes to a temp path then atomically renames.

Initial tool list (MVP and beyond):

| Category | Tool | Tier | Notes |
|---|---|---|---|
| Index | `index.build` | mutate | Wraps `pyvsag.Index.build` |
| Index | `index.add` / `index.remove` | mutate / destructive | |
| Index | `index.search` (knn / range) | read | |
| Index | `index.save` / `index.load` | mutate / read | |
| Index | `index.merge` / `index.clone` | mutate | |
| Index | `index.stats` | read | `GetNumElements`, memory, etc. |
| Index | `index.analyze` | read | wraps `tools/analyze_index` |
| Index | `index.tune` | mutate | high-level, owns the bench loop |
| Index | `index.estimate_memory` | read | |
| Index | `index.export_model` | mutate | |
| Dataset | `dataset.list` / `dataset.info` / `dataset.peek` | read | |
| Dataset | `dataset.download` | mutate | wraps `download_annbench_datasets.sh` |
| Bench | `bench.run` | read | wraps `tools/eval/eval_performance` |
| Bench | `bench.compare` | read | takes N result files → pareto / table |
| Params | `params.preset.list` / `.show` / `.use` | read | from `benchs/README.md` recipes |
| Params | `params.validate` | read | JSON-schema check before build |
| Dev | `dev.test` / `dev.fmt` / `dev.lint` | mutate | wraps `make` |
| Docs | `docs.search` | read | RAG over examples + docs + headers |
| Examples | `examples.find` / `examples.show` | read | |
| Session | `todo.add` / `.update` / `.list` | read/mutate | mirrored in TUI panel |
| Session | `task.spawn` / `.status` / `.attach` | mutate | background long ops |
| Shell | `bash.run` (sandboxed, allowlist) | mutate | escape hatch |

### 3.6 What is unique vs. a generic coding agent

1. **Operations are reversible** — `clone` before mutate, snapshot params. LLM can experiment more aggressively.
2. **Operations have quantitative feedback** — recall / QPS / latency / memory close the loop without a human in the middle. ReAct-style autonomous tuning is genuinely useful here.
3. **Structured knowledge base** — `examples/` (37 files), `benchs/README.md` (recipe table), `include/vsag/` (API schema), `docs/`. A small RAG store dramatically beats a generic LLM.
4. **Long-running tasks are the norm** — 30-min builds, multi-hour benchmark sweeps. Background tasks + persistent todos are core, not garnish.
5. **Multimodal terminal output is high-value** — recall–QPS Pareto curves, parameter comparison tables, degree distributions, connected-component summaries.

### 3.7 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Long-op UX (build/bench take tens of minutes) | Background task system; SQLite-persisted job queue; TUI progress panel; chat remains usable while job runs |
| Large data context (datasets are GB-scale) | Only metadata (shape/dtype/stats/head) goes into prompts; data accessed via tools |
| LLMs don't know VSAG vocabulary | RAG over `examples/` + `benchs/README.md` + headers; few-shot examples in system prompt |
| Tool granularity (too coarse → uncontrollable; too fine → token blowup) | Two-tier design (§3.5) |
| Error recovery | Structured `{code, message, suggestion}` returns; system-prompt section on common errors |
| Safety (delete index by mistake) | Three-tier permissions; destructive ops require explicit confirm; full audit log |
| LLM cost / latency | Provider-agnostic; allow local OpenAI-compatible endpoints (Ollama / vLLM / DeepSeek) |
| Maintenance drift vs. core VSAG | Build only on top of `pyvsag` + stable subprocess interfaces; pin versions |

---

## 4. Implementation Plan

### 4.1 Recommended order of work

1. **Spike (½–1 day)** — bare Python script, no TUI, hand-built tool calls, OpenAI function-calling, 3 tools: `index.build`, `index.search`, `bench.run`. Goal: validate the LLM can actually drive VSAG with our schemas. **Do this before any UI work.**
2. **Tool schema design (1–2 days)** — write JSON Schemas + permission tiers + error contracts for all MVP tools. This is the spec the rest of the agent depends on.
3. **MVP build (1–2 weeks)** — Textual TUI + Agent loop + 8 core tools + RAG + background tasks.
4. **Polish & release (3–5 days)** — packaging, docs, demo recording, config UX.

### 4.2 MVP scope (2–3 weeks)

**Week 1 — foundation**
- Textual TUI skeleton (chat stream + tool-call cards + input)
- LLM provider abstraction (OpenAI-compatible first, Anthropic adapter)
- 8 core tools: `index.build`, `index.search`, `index.stats`, `index.analyze`, `bench.run`, `dataset.list`, `dataset.info`, `params.preset.show`
- 3-tier permission system + confirm prompts

**Week 2 — intelligence**
- RAG ingest of `examples/`, `benchs/README.md`, `include/vsag/*.h`, `docs/`
- Plan/Todo system reflected in TUI panel
- Background task runner + progress
- Session persistence (SQLite) + `--resume`

**Week 3 — UX**
- Tables / charts (recall–QPS Pareto in terminal; e.g. `plotext` or Textual's plot widgets)
- Robust error handling + retries
- Config (`~/.vsag-code/config.toml`)
- Packaging (`pip install vsag-code` and/or `uvx vsag-code`)

### 4.3 Repository layout

Put the agent inside the VSAG repo (recommended) under a new top-level `cli/` (the name `cli/` matches the parent dir already used outside the repo, but inside the repo we should use a clearer name — proposed: **`agent/`** to avoid confusion with a CLI subcommand tree):

```
vsag/
└── agent/                      # vsag-code lives here (Python, isolated build)
    ├── pyproject.toml
    ├── README.md
    ├── vsag_code/
    │   ├── __init__.py
    │   ├── __main__.py         # entry: `python -m vsag_code` / `vsag-code`
    │   ├── tui/                # Textual app
    │   ├── agent/              # loop, prompts, planning, todo
    │   ├── llm/                # provider adapters
    │   ├── tools/              # one file per tool category
    │   │   ├── index_tools.py
    │   │   ├── dataset_tools.py
    │   │   ├── bench_tools.py
    │   │   ├── params_tools.py
    │   │   ├── dev_tools.py
    │   │   ├── docs_tools.py
    │   │   └── shell_tools.py
    │   ├── rag/                # ingest + retrieval
    │   ├── tasks/              # background task runner
    │   ├── config.py
    │   └── session.py          # SQLite persistence
    └── tests/
```

Rationale for in-repo: keeps RAG ingest paths stable, makes CI integration trivial, single source of truth. If maintainers prefer a separate repo, the layout still works as `vsag-code/` standalone.

### 4.4 Tech stack and rationale

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | `pyvsag` is Python; ecosystem (numpy/h5py/pandas) is native; LLM SDKs first-class |
| TUI | [Textual](https://textual.textualize.io/) | Async, rich widgets, tables, progress, scrollback, panels; mature |
| LLM SDK | `openai` (any OpenAI-compatible base_url) + `anthropic` adapter | Covers OpenAI / DeepSeek / Qwen / Ollama / vLLM in one shot |
| Tool schema | Pydantic v2 → JSON Schema | Single source for validation + LLM function schema |
| RAG | Local vector store (e.g. `chromadb` or `sqlite-vss`); embeddings via provider or local | Small dataset (~few MB of text), no infra needed |
| Background tasks | `asyncio` + a process pool for blocking VSAG calls | `pyvsag` calls release GIL during heavy ops; subprocess for `eval_performance` |
| Persistence | SQLite (`sqlite3` stdlib) | Sessions, tasks, audit log |
| Plotting (terminal) | `plotext` or Textual's `Plot` | No GUI dep |
| Packaging | `uv` / `hatchling`; entry point `vsag-code` | Modern Python tooling |

---

## 5. Resume Kit (read this first when picking up)

If you are an engineer picking this up in a long-running environment, here is everything you need:

### 5.1 Branch and remote

- Branch: `feature/vsag-code-tui-agent` (this branch).
- Pushed to fork: `git@github.com:wxyucs/vsag.git`.
- Upstream: `git@github.com:antgroup/vsag.git` (kept as `origin` — do NOT push agent work here without a PR).

### 5.2 First steps to validate the idea (do before touching the TUI)

```bash
# 1. Build pyvsag locally
cd vsag
make pyvsag PY_VERSION=3.10

# 2. Build the C++ tools you'll wrap
make dev      # turns on ENABLE_TOOLS, ENABLE_TESTS, ENABLE_PYBINDS, ENABLE_EXAMPLES

# 3. Confirm the wrappable surfaces work standalone
./build-debug/tools/eval/eval_performance --help
./build-debug/tools/analyze_index/analyze_index --help

# 4. (Optional) Get a small dataset
./scripts/download_annbench_datasets.sh sift  # or similar

# 5. Spike: write ~150 lines of Python that:
#      - takes a hard-coded user goal
#      - exposes 3 tools (build/search/bench) as OpenAI function specs
#      - runs the LLM loop until done
#      - prints to stdout (no Textual yet)
#    Goal: prove the LLM can actually call these tools coherently.
```

If the spike works, proceed with the MVP plan in §4.2. If it doesn't (LLM hallucinates params, can't recover from errors), iterate on **tool schemas and system prompt** before investing in TUI.

### 5.3 Key files to read in this order

1. `README.md` — what VSAG is
2. `examples/cpp/103_hgraph.cpp` and `examples/python/103_hgraph.py` — canonical usage
3. `benchs/README.md` — parameter recipes (becomes the LLM's cheat sheet)
4. `tools/eval/main.cpp` — see how params flow through today
5. `include/vsag/factory.h` and `include/vsag/index.h` — the API the agent ultimately drives
6. `python/pyvsag/__init__.py` — what's actually exposed to Python
7. This document, §3.5 — the tool list to implement

### 5.4 What is intentionally NOT decided yet

See §6.

### 5.5 Things to NOT do

- Do NOT change C++ source for the agent's sake. The agent rides on stable surfaces.
- Do NOT mix the agent's Python deps into the existing `python/` (which is the binding). Keep `agent/` isolated with its own `pyproject.toml`.
- Do NOT push the agent to `origin` (upstream); push only to the `fork` remote until a PR is reviewed.
- Do NOT skip the spike (§5.2). Tool-calling quality is the whole game; learn it cheaply first.

---

## 6. Open Questions

1. **Repo placement** — `agent/` inside `antgroup/vsag` vs. separate `wxyucs/vsag-code` repo? In-repo is recommended (§4.3) but maintainers may prefer separation.
2. **Default LLM provider** — OpenAI? DeepSeek (cost)? Local Ollama (privacy)? MVP supports all; default is a UX call.
3. **High-level vs. low-level tools** — exactly which goals deserve a high-level tool? Initial guess: `index.tune_for_targets(recall, p99)`, `bench.compare`, `index.build_from_preset`. Refine after the spike.
4. **Embeddings for RAG** — provider-hosted (cost) vs. local sentence-transformers (extra dep). Probably local for MVP.
5. **Permission UX** — global "always allow `read`", per-tool overrides, `--yolo` mode? Borrow from Claude Code / OpenCode conventions.
6. **i18n** — TUI labels in English by default, but VSAG has many CN-speaking users. Probably ship CN strings + `--lang` switch.
7. **Telemetry** — opt-in usage metrics to improve the agent? Sensitive; default off.

---

## Appendix A. Concrete VSAG facts referenced by the agent

A condensed cheat-sheet the agent's system prompt / RAG should always have access to.

### A.1 Algorithm × goal recipes (extracted from `benchs/README.md` shape)

> **Note**: read `benchs/README.md` directly when implementing — these are placeholders showing the *shape* the RAG should retrieve, not authoritative recipes:

```yaml
- algorithm: hgraph
  recall_target: 0.95
  dataset_class: dense_768d
  create_params:
    base_quantization_type: sq8_uniform
    max_degree: 32
    ef_construction: 400
  search_params:
    ef_search: 60
```

The agent should not hard-code these; it should retrieve them from the actual `benchs/README.md`.

### A.2 JSON parameter shapes

Already documented in §2.8. Reproduced here for quick reference:

- HNSW (flat): `{"dim", "dtype", "metric_type", "hnsw": {"max_degree", "ef_construction"}}`
- HGraph / IVF / Pyramid (wrapped): `{"dim", "dtype", "metric_type", "index_param": {...}}`
- Search: `{"<algo>": {"ef_search"|"scan_buckets_count"|"beam_search"|"io_limit"|...}}`

### A.3 Existing tool invocations (to wrap, not reimplement)

```bash
# Single-test
./tools/eval/eval_performance \
  -d sift.hdf5 -t build -n hgraph \
  -c '{"dim":128,...}' -i out.idx

./tools/eval/eval_performance \
  -d sift.hdf5 -t search -n hgraph \
  -i out.idx -s '{"hgraph":{"ef_search":60}}' \
  --search_mode knn --topk 10

# Batch
./tools/eval/eval_performance --config benchs/datasets/sift-90.yml

# Diagnostics
./tools/analyze_index/analyze_index \
  -i out.idx -bp '{...}' -qp queries.hdf5 -sp '{...}' -k 10
```

---

## Appendix B. Verbatim conversation log

For full provenance and to preserve nuance the summary above may have lost.

---

### B.1 User turn 1

> vsag/是我们的项目，你可以通过本地代码，或者GitHub网站，或者我们的项目官网来了解这个项目。CLI 概念现在很火，我需要你分析当前项目所具备的工具/调试/测试能力，然后告诉我，是否可以为VSAG项目开发一个 VSAG CLI 工具（例如把现有的能力都包进去，甚至支持配置LLM API，支持自然语言来操作这个library）。

### B.2 Assistant turn 1 (summary of the long reply — full content captured in §2 and §1.1)

The assistant explored the repo (tools/, benchs/, tests/, examples/, bindings/, scripts/, include/vsag/), inventoried capabilities, and concluded that a CLI is feasible. It proposed a two-phase plan: Phase 1 — wrap existing tools as a subcommand tree (`vsag build/search/bench/analyze/...`); Phase 2 — add a `vsag ai "..."` natural-language layer using LLM function calling, with the CLI subcommands as the function set, RAG over `examples/` and `benchs/README.md`, and dry-run-by-default safety. Recommended language: Python (`pyvsag` + Click/Typer). MVP: ~6 commands, 1–2 weeks. Asked which next step to take (design doc / scaffold / LLM layer / RFC).

### B.3 User turn 2

> 不错，但是对于CLI的理解不太一样。我想要一个有TUI的工具，就像OpenCode这样对话式的

### B.4 Assistant turn 2 (summary — full content is §3 of this document)

The assistant reframed: the user wants a conversational TUI agent (OpenCode-style), not a CLI subcommand tree. It produced:

- A mapping table from OpenCode primitives (Read/Write/Bash/Grep/Todo) to VSAG primitives (Build/Add/Search/Bench/Tune/Plan).
- Two example interaction transcripts (goal-driven build with confirmations; index diagnosis).
- A four-layer architecture (TUI / Agent loop / Tool registry / VSAG library).
- A categorized tool registry (index / dataset / bench / params / dev / docs / shell) with 3-tier permissions.
- Five differentiators vs. a generic coding agent: reversibility, quantitative feedback, structured knowledge, long-running tasks, multimodal terminal output.
- Risk table (long-op UX, large data, LLM domain knowledge, tool granularity, error recovery, safety) with mitigations.
- A 2–3-week MVP plan in three weekly tranches (foundation / intelligence / UX).
- Strong recommendation to **do a non-TUI spike first** to validate LLM tool-calling on this domain before investing in Textual.
- Final pitch: "It does not compete with OpenCode — they compose. Use OpenCode to edit VSAG source, use VSAG-Code to drive VSAG indexes."

Recommended sequence: spike → tool-schema design → MVP scaffold.

### B.5 User turn 3

> 把你的方案，以及和我的对话写到本地的markdown文件，然后提交到我forked的仓库中的一个分支上。保存好上下文，我会在一个可长期运行的环境中完成这个任务的开发。

### B.6 Assistant turn 3

This document. Plus: created branch `feature/vsag-code-tui-agent`, added `wxyucs/vsag` as `fork` remote, committed this file, pushed to the fork.

---

*End of proposal. The agent picking this up should start at [§5 Resume Kit](#5-resume-kit-read-this-first-when-picking-up).*
