# Distributed under the MIT License.
# See LICENSE.txt for details.

from dataclasses import dataclass
from pathlib import Path


BIN_DIR_NAME = "bin"


@dataclass(frozen=True)
class BinDirectory:
    path: Path

    @classmethod
    def this(cls) -> "BinDirectory":
        """The bin directory that this script runs from"""
        # This file sits in bin/python/spectre/support/ (three levels deep).
        # Don't resolve symlinks, so this works also with PY_DEV_MODE.
        return cls(path=Path(__file__).parents[3])
