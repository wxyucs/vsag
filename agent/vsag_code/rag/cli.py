#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""``python -m vsag_code.rag.cli`` -- ingest / inspect helpers.

Examples::

    # Build the store at $VSAG_CODE_HOME/rag from the local checkout.
    python -m vsag_code.rag.cli ingest --repo /workspace/vsag

    # Spot-check retrieval without going through an LLM.
    python -m vsag_code.rag.cli search --query "how do I tune ef_search?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .ingest import ingest_repo
from .retrieve import retrieve
from .store import DEFAULT_STORE_DIR


def _store_dir(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("VSAG_CODE_RAG_DIR") or DEFAULT_STORE_DIR)


def _cmd_ingest(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser()
    if not repo.is_dir():
        print(f"error: repo not found: {repo}", file=sys.stderr)
        return 2
    store_dir = _store_dir(args.store_dir)
    store = ingest_repo(repo, store_dir)
    print(
        f"ingested {len(store.chunks)} chunks from {repo} -> {store_dir}",
        file=sys.stderr,
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    hits = retrieve(args.query, k=args.k, store_dir=str(_store_dir(args.store_dir)))
    json.dump({"query": args.query, "hits": hits}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="vsag_code.rag.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Build / refresh the doc store.")
    ing.add_argument("--repo", required=True, help="Path to the VSAG checkout.")
    ing.add_argument("--store-dir", default=None, help="Override store path.")
    ing.set_defaults(func=_cmd_ingest)

    srch = sub.add_parser("search", help="Spot-check retrieval.")
    srch.add_argument("--query", required=True)
    srch.add_argument("--k", type=int, default=5)
    srch.add_argument("--store-dir", default=None)
    srch.set_defaults(func=_cmd_search)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
