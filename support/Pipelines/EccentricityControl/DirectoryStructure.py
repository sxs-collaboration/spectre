# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Name of the directory that holds the initial data of an eccentricity-control
# iteration
ID_DIR_NAME = "ID"


@dataclass(frozen=True, order=True)
class EccIteration:
    """One iteration of the eccentricity-control loop

    Each iteration generates initial data, evolves it for a few orbits, and
    measures the eccentricity. If the eccentricity is not yet within tolerance,
    the next iteration starts with updated orbital parameters. Once it is, the
    evolution of the last iteration continues to completion, branching into one
    or more resolutions ("Levs").

    We currently write eccentricity-control iterations in a directory structure
    like this:

    ```
    PIPELINE_DIR/
        Ecc0/
            ID/
                0000_InitialData/
                0001_InitialData/
                ...
            Lev1/
                0000_Inspiral/
                0001_Inspiral/
                ...
        Ecc1/
            ...
    ```

    The 'ID' directory and the 'Lev' directories are segments directories, so
    the runs in them are 'Segment's (see
    'spectre.support.DirectoryStructure.Segment').

    WARNING: Don't assume that simulations always have the above directory
    structure. You don't want your code to break when files are copied, moved
    around, or renamed. Instead of relying on some directory structure, have
    your code take the files it needs as input. This is quite easy using globs.
    """

    path: Path
    id: int

    NAME_PATTERN = re.compile(r"Ecc(\d+)")

    @classmethod
    def match(cls, path: Union[str, Path]) -> Optional["EccIteration"]:
        """Checks if the 'path' is an eccentricity-control iteration"""
        path = Path(path)
        match = re.match(cls.NAME_PATTERN, path.resolve().name)
        if not match:
            return None
        return cls(path=path, id=int(match.group(1)))

    @classmethod
    def last(cls, pipeline_dir: Union[str, Path]) -> Optional["EccIteration"]:
        """The last iteration in the 'pipeline_dir', or 'None' if it has none"""
        all_iterations = list_ecc_iterations(pipeline_dir)
        return all_iterations[-1] if all_iterations else None

    @classmethod
    def next(cls, pipeline_dir: Union[str, Path]) -> "EccIteration":
        """The next iteration to create in the 'pipeline_dir'

        Continues the numbering of the iterations in the 'pipeline_dir', or
        starts at zero if it has none.
        """
        last_iteration = cls.last(pipeline_dir)
        next_id = last_iteration.id + 1 if last_iteration else 0
        return cls(path=Path(pipeline_dir) / f"Ecc{next_id}", id=next_id)

    @classmethod
    def current(cls, pipeline_dir: Union[str, Path]) -> "EccIteration":
        """The iteration in progress in the 'pipeline_dir'

        The first iteration if the 'pipeline_dir' has none yet, or else the last
        iteration.
        """
        return cls.last(pipeline_dir) or cls.next(pipeline_dir)

    @property
    def id_dir(self) -> Path:
        """Directory that holds the initial data of this iteration"""
        return self.path / ID_DIR_NAME

    def lev_dir(self, lev: int) -> Path:
        """Directory that holds the evolution of this iteration at 'lev'"""
        return self.path / f"Lev{lev}"


def list_ecc_iterations(pipeline_dir: Union[str, Path]) -> List[EccIteration]:
    """All eccentricity-control iterations in the 'pipeline_dir'"""
    pipeline_dir = Path(pipeline_dir)
    if not pipeline_dir.exists():
        return []
    matches = map(EccIteration.match, pipeline_dir.iterdir())
    # Sort by id, not by name, because the ids are not zero-padded
    return sorted(
        (match for match in matches if match),
        key=lambda iteration: iteration.id,
    )
