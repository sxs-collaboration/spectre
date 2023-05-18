# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Checkpoint:
    """State of a simulation saved to disk

    An executable can write multiple checkpoints during its execution, and can
    be restarted from those checkpoints. The 'id' enumerates the checkpoints.

    We currently write checkpoints in a directory structure like this:

    ```
    RUN_DIR/
        Checkpoints/
            Checkpoint_0000/
            Checkpoint_0001/
            ...
    ```

    WARNING: Don't assume checkpoints always exist in the above directory
    structure. You don't want your code to break when checkpoints are copied,
    moved around, or renamed.
    """

    path: Path
    id: int

    NAME_PATTERN = re.compile(r"Checkpoint_(\d+)")
    NUM_DIGITS = 4

    @classmethod
    def match(cls, path: Union[str, Path]) -> Optional["Checkpoint"]:
        """Checks if the 'path' is a checkpoint"""
        path = Path(path)
        match = re.match(cls.NAME_PATTERN, path.resolve().name)
        if not match:
            return None
        return cls(path=path, id=int(match.group(1)))


def list_checkpoints(checkpoints_dir: Union[str, Path]) -> List[Checkpoint]:
    """All checkpoints in the 'checkpoints_dir'"""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return []
    matches = map(Checkpoint.match, checkpoints_dir.iterdir())
    return sorted(match for match in matches if match)


@dataclass(frozen=True, order=True)
class Segment:
    """Part of a simulation that ran as one executable invocation

    We have to split simulations into segments because supercomputers don't
    typically support unlimited run times. Therefore, we terminate the job,
    write the simulation state to disk as a checkpoint, and submit a new job
    that restarts from the last checkpoint.

    We currently write segments in a directory structure like this:

    ```
    SEGMENTS_DIR/
        Segment_0000/
            InputFile.yaml
            Submit.sh
            Output.h5
            Checkpoints/
        Segment_0001/...
    ```

    WARNING: Don't assume that simulations always have the above directory
    structure. You don't want your code to break when files are copied, moved
    around, or renamed. Instead of relying on some directory structure, have
    your code take the files it needs as input. This is quite easy using globs.
    """

    path: Path
    id: int

    NAME_PATTERN = re.compile(r"Segment_(\d+)")
    NUM_DIGITS = 4

    @classmethod
    def first(cls, directory: Path) -> "Segment":
        name = "Segment_" + "0" * cls.NUM_DIGITS
        return Segment(path=directory / name, id=0)

    @property
    def next(self) -> "Segment":
        next_id = self.id + 1
        next_name = "Segment_" + str(next_id).zfill(self.NUM_DIGITS)
        return Segment(path=self.path.resolve().parent / next_name, id=next_id)

    @classmethod
    def match(cls, path: Union[str, Path]) -> Optional["Segment"]:
        """Checks if the 'path' is a segment"""
        path = Path(path)
        match = re.match(cls.NAME_PATTERN, path.resolve().name)
        if not match:
            return None
        return cls(path=path, id=int(match.group(1)))

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "Checkpoints"

    @property
    def checkpoints(self) -> List[Checkpoint]:
        return list_checkpoints(self.checkpoints_dir)


def list_segments(segments_dir: Union[str, Path]) -> List[Segment]:
    """All segments in the 'segments_dir'"""
    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        return []
    matches = map(Segment.match, segments_dir.iterdir())
    return sorted(match for match in matches if match)
