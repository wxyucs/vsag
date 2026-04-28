# vsag/agent — VSAG-Code conversational agent

Implementation home for the VSAG-Code TUI agent described in
[`../VSAG_CODE_PROPOSAL.md`](../VSAG_CODE_PROPOSAL.md).

## Layout

```
agent/
├── README.md              this file
├── spike/                 stage 1: ~150-line LLM tool-calling spike
│   ├── spike.py           single-script driver
│   ├── tools.py           tool implementations (no LLM dep)
│   └── RESULTS.md         provider-by-provider trace + verdict
├── tools/                 stage 2+: schemas + production tool impls
└── (vsag_code/, tests/, etc. follow in later stages)
```

## Quick run (stage 1 spike)

The spike runs entirely inside the `vsag-dev` pod (see
`HouseBrain/k8s/apps/vsag-dev/`). It assumes:

- VSAG built via `make dev` at `/workspace/vsag/build/`
- `_pyvsag.cpython-310-*.so` discoverable through `PYTHONPATH`
- SIFT-128 dataset at `/data/datasets/sift-128-euclidean.hdf5`

```bash
# Inside the pod (env primed by /etc/profile.d/zz-vsag-dev.sh):
cd /workspace/vsag
python3 agent/spike/spike.py --provider copilot --goal "build hgraph index on SIFT and report recall@10"
```

Provider list and gating criteria: see proposal §5.2.

## Status

- [x] Stage -1: K8s dev pod (HouseBrain `feat(k8s): add vsag-dev`)
- [x] Stage 0: pyvsag built, SIFT downloaded, smoke import OK
- [ ] Stage 1: spike (in progress)
- [ ] Stage 2: tool schemas
- [ ] Stage 3: MVP TUI
- [ ] Stage 4: RAG over benchs/
- [ ] Stage 5: UX polish
