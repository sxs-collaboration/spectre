# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Directories that simulations are organized in

Three kinds of directory appear throughout the code, and the name of a
variable says which kind it holds:

- 'run_dir': one invocation of an executable runs here. It holds the input
  file, the submit script, output files, and a 'Checkpoints' directory.
- 'segments_dir': holds a numbered sequence of runs, such as '0000_Inspiral'
  through '0013_Ringdown'. See 'Segment' below.
- 'pipeline_dir': the directory of a simulation, in which a pipeline creates
  all its runs. Each pipeline defines how it groups runs in there, e.g.
  'spectre.Pipelines.EccentricityControl.DirectoryStructure'.

A name that qualifies one of these, such as 'id_run_dir' or
'inspiral_run_dir', refers to that directory of a specific step.
"""

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
    that restarts from the last checkpoint. Segments are also how a pipeline
    proceeds from one executable to the next, e.g. from an inspiral to a
    ringdown. Therefore, each segment carries a 'label' that says what ran in
    it, in addition to the 'id' that enumerates the segments.

    We currently write segments in a directory structure like this:

    ```
    SEGMENTS_DIR/
        0000_Inspiral/
            Inspiral.yaml
            Submit.sh
            Output.h5
            Checkpoints/
        0001_Inspiral/
            ...
        0002_Ringdown/
            ...
    ```

    Note: "Inspiral" and "Ringdown" are examples, any label can be used. The id
    goes _before_ the label so that a plain 'ls' of the segments directory lists
    the segments in the order they ran, no matter which executable ran in them.

    WARNING: Don't assume that simulations always have the above directory
    structure. You don't want your code to break when files are copied, moved
    around, or renamed. Instead of relying on some directory structure, have
    your code take the files it needs as input. This is quite easy using globs.
    """

    path: Path
    id: int
    label: str

    NAME_PATTERN = re.compile(r"(\d+)_(.+)")
    NUM_DIGITS = 4

    @classmethod
    def match(cls, path: Union[str, Path]) -> Optional["Segment"]:
        """Checks if the 'path' is a segment"""
        path = Path(path)
        match = re.match(cls.NAME_PATTERN, path.resolve().name)
        if not match:
            return None
        return cls(path=path, id=int(match.group(1)), label=match.group(2))

    @classmethod
    def last(cls, segments_dir: Union[str, Path]) -> Optional["Segment"]:
        """The last segment in the 'segments_dir', or 'None' if it has none"""
        all_segments = list_segments(segments_dir)
        return all_segments[-1] if all_segments else None

    @classmethod
    def next(cls, segments_dir: Union[str, Path], label: str) -> "Segment":
        """The next segment to create in the 'segments_dir'

        Continues the numbering of the segments in the 'segments_dir', or starts
        at zero if it has none.
        """
        last_segment = cls.last(segments_dir)
        next_id = last_segment.id + 1 if last_segment else 0
        next_name = f"{str(next_id).zfill(cls.NUM_DIGITS)}_{label}"
        return cls(path=Path(segments_dir) / next_name, id=next_id, label=label)

    @property
    def input_file(self) -> Path:
        """The input file for the segment (has the same name as the label)"""
        return self.path / f"{self.label}.yaml"

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
    return sorted(
        (match for match in matches if match), key=lambda segment: segment.id
    )
