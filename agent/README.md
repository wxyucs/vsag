# vsag/agent — VSAG-Code conversational agent

Implementation of the VSAG-Code TUI agent described in
[`../VSAG_CODE_PROPOSAL.md`](../VSAG_CODE_PROPOSAL.md).

VSAG-Code is a non-coding agent: the LLM never edits VSAG sources.
Instead it operates the library by calling a curated, schema-validated
set of Python tools (built on top of `pyvsag`), so a user can say
*"build an HGraph index on SIFT-1M and tell me recall@10 at ef_search=100"*
and get a grounded, numeric answer in one turn.

## Layout

```
agent/
├── README.md              this file
├── pyproject.toml         hatchling package, console script `vsag-code`
├── spike/                 stage 1: ~150-line LLM tool-calling spike
│   ├── spike.py           single-script driver
│   ├── tools.py           tool implementations (no LLM dep)
│   └── RESULTS.md         provider-by-provider trace + verdict
├── vsag_code/             stage 2+: package
│   ├── __main__.py        CLI: `vsag-code`
│   ├── tools/             schemas, registry, dataset/index tools
│   ├── llm/               provider client (openai/deepseek/copilot/anthropic/ollama)
│   ├── agent/             system prompts (EN+ZH), tool-calling loop
│   ├── tui/               prompt_toolkit REPL + event renderer
│   └── rag/               doc ingest, embedding store, docs_search tool
└── tests/                 pytest suite (no network, no pyvsag required)
```

## Install

```bash
# Inside the vsag-dev pod, with pyvsag + h5py + numpy already on PYTHONPATH.
pip install -e ./agent             # core only
pip install -e './agent[rag]'      # + sentence-transformers for docs_search
pip install -e './agent[dev]'      # + pytest
```

## Quick run

```bash
# One-shot, headless. Streams events to stdout, writes full JSON trace.
vsag-code --provider deepseek --once \
  "Inspect /data/datasets/sift-128-euclidean.hdf5, build an HGraph index on the
   full training set, and report recall@10 with QPS at ef_search=100." \
  --out /tmp/trace.json

# Interactive REPL.
vsag-code --provider copilot
```

CLI options:

```
--provider {openai,deepseek,copilot,anthropic,ollama}
--model MODEL          override default model id
--lang {en,zh}         system-prompt locale
--once GOAL            run one goal headlessly and exit
--out PATH             dump JSON trace (one-shot mode)
--yes                  auto-allow destructive tools
--max-steps N          loop iteration cap (default 16)
--no-color             disable ANSI colors
--history PATH         override REPL history file
```

REPL slash commands: `/help`, `/provider`, `/reset`, `/trace [PATH]`, `/quit`.

## Tools the agent can call

Tier `read` (no confirmation):

- `dataset_list(prefix?)` -- list HDF5 datasets under `$VSAG_DATASETS`
- `dataset_info(path)` -- shape, dim, dtype, distance metric
- `dataset_peek(path, n?, partition?)` -- head-slice, capped at 16 rows
- `index_search(handle, topk?, ef_search?, num_queries?)` -- KNN with
  recall@k + mean / p50 / p95 / p99 latency + QPS
- `index_stats(handle)` -- metadata + coarse memory estimate
- `index_list()` -- enumerate handles (in-memory + on-disk)
- `docs_search(query, k?)` -- RAG over public headers, examples, docs

Tier `mutate`:

- `index_build(dataset_path, algorithm?, metric?, index_param?, num_elements?)`
  -- returns an opaque handle (e.g. `hgraph-idx-1`) the LLM reuses across turns

Tier `destructive` (requires explicit user confirm; auto-denied otherwise):

- `index_remove(handle)`

Tools never raise into the loop: failures come back as
`{"error": {"code", "message", "suggestion"}}` with codes
`not_found / invalid_argument / unsupported / permission_denied / internal`.

## RAG (optional)

Build the local doc store once, then the agent can call `docs_search`:

```bash
python -m vsag_code.rag.cli ingest --repo /workspace/vsag
python -m vsag_code.rag.cli search --query "how do I tune ef_search"
```

Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, normalized,
stored as `manifest.json` + `embeddings.npy` under
`$VSAG_CODE_HOME/rag` (default `/workspace/.vsag-code/rag`).

## Environment

```
VSAG_DATASETS         dataset root (default /data/datasets)
VSAG_CODE_HOME        agent state root (default /workspace/.vsag-code)
VSAG_CODE_RAG_DIR     override RAG store path
VSAG_CODE_PROVIDER    default --provider
VSAG_CODE_MODEL       default --model
VSAG_CODE_LANG        default --lang

OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
ANTHROPIC_API_KEY / ANTHROPIC_MODEL
OLLAMA_BASE_URL / OLLAMA_MODEL
COPILOT_MODEL                            # github token cached at $VSAG_CODE_HOME
HTTPS_PROXY                              # used automatically by urllib
```

## Tests

```bash
python -m pytest agent/tests -q
```

The suite avoids real LLM and real `pyvsag` calls (they are mocked), so
it runs on macOS / CI without a built VSAG binary. End-to-end smoke
runs happen inside the `vsag-dev` pod -- see `RESULTS.md` for the
recall@10 / QPS gate the agent must clear before each release.

## Status

- [x] Stage -1: K8s dev pod (HouseBrain `feat(k8s): add vsag-dev`)
- [x] Stage 0: pyvsag built, SIFT downloaded, smoke import OK
- [x] Stage 1: spike (DeepSeek + Copilot, recall@10 within 0.2pp of baseline)
- [x] Stage 2: tool schemas, registry, agent loop, LLM clients
- [x] Stage 3: CLI + prompt_toolkit REPL
- [x] Stage 4: RAG over docs/ + headers + examples
- [x] Stage 5: tests, README, polish
