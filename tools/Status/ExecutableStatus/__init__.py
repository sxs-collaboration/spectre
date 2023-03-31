# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re

from .EvolveGhBinaryBlackHole import EvolveGhBinaryBlackHole
from .ExecutableStatus import ExecutableStatus, EvolutionStatus

logger = logging.getLogger(__name__)

# Subclasses are matched in this order. Add new subclasses here.
executable_status_subclasses = [
    EvolveGhBinaryBlackHole,
    EvolutionStatus,
]


def match_executable_status(executable_name: str) -> ExecutableStatus:
    """Get the 'ExecutableStatus' subclass that matches the 'executable_name'.

    The 'executable_name' is matched against all 'executable_name_patterns' in
    the 'executable_status_subclasses' list. If no pattern matches, a plain
    'ExecutableStatus' object is returned.
    """
    for cls in executable_status_subclasses:
        for pattern in cls.executable_name_patterns:
            if re.match(pattern, executable_name):
                return cls()
    logger.warning("No 'ExecutableStatus' subclass matches executable name "
                   f"'{executable_name}'. Implement one and add it to "
                   "'executable_status_classes'.")
    return ExecutableStatus()
