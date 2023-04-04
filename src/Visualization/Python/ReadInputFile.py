# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Tools for parsing YAML input files

To simply read an input file as a dictionary, use a standard YAML parser like
PyYAML or ruaml.yaml:

    import yaml
    with open(input_file_path) as open_input_file:
        input_file = yaml.safe_load(open_input_file)

The functions in this module provide additional functionality to work with input
files.
"""

import re
from typing import Optional


def get_executable(input_file_content: str) -> Optional[str]:
    """Extract the executable the input file is supposed to run with.

    The executable is parsed from a comment like this in the input file:

        # Executable: EvolveScalarWave

    Arguments:
      input_file_contents: The full input file read in as a string.

    Returns: The executable ("EvolveScalarWave" in the example above), or None
      if no executable was found.
    """
    match = re.search(r'#\s+Executable:\s+(.+)', input_file_content)
    if not match:
        return None
    return match.group(1)


def find_event(event_name: str, input_file: dict) -> dict:
    """Find a particular event in the "EventsAndTriggers" of the input file.

    Arguments:
      event_name: The name of an event like "ObserveTimeSteps".
      input_file: The input file read in as a dictionary.

    Returns: The event as a dictionary, or None if the event wasn't found.
    """
    for _, events in input_file["EventsAndTriggers"]:
        for event in events:
            if event_name in event:
                return event[event_name]
    return None
