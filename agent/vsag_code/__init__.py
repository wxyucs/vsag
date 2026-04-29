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

"""VSAG-Code: a conversational TUI agent for the VSAG vector index library.

The package is organized into small layers that can be exercised without
each other:

* ``vsag_code.tools``   pure-Python tools the LLM is allowed to call
* ``vsag_code.llm``     provider-agnostic chat-completions client
* ``vsag_code.agent``   the tool-calling loop + system prompt + permissions
* ``vsag_code.rag``     local retrieval over VSAG docs / examples / headers
* ``vsag_code.tui``     interactive REPL on top of the agent loop
* ``vsag_code.session`` persistence of indexes + transcripts

The CLI entry-point is :mod:`vsag_code.__main__` (registered as the
``vsag-code`` script in ``pyproject.toml``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
