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

"""Pytest fixtures shared across the agent test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``vsag_code`` importable when the package is not pip-installed.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


@pytest.fixture
def vsag_code_home(tmp_path, monkeypatch):
    """Redirect ``VSAG_CODE_HOME`` to a tmpdir and reset the registry."""
    monkeypatch.setenv("VSAG_CODE_HOME", str(tmp_path))
    from vsag_code.tools.registry_index import reset_registry_for_tests

    reset_registry_for_tests()
    yield tmp_path
    reset_registry_for_tests()
