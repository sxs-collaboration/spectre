# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Tools for parsing YAML input files

To simply read an input file as a dictionary, use a standard YAML parser like
PyYAML or ruaml.yaml:

    import yaml
    with open(input_file_path) as open_input_file:
        metadata, input_file = yaml.safe_load_all(open_input_file)

It's also possible to load just the metadata without the overhead of parsing the
full input file:

    with open(input_file_path) as open_input_file:
        metadata = next(yaml.safe_load_all(open_input_file))

The functions in this module provide additional functionality to work with input
files.
"""

import re
from typing import Optional


def find_event(event_name: str, input_file: dict) -> dict:
    """Find a particular event in the "EventsAndTriggers" of the input file.

    Arguments:
      event_name: The name of an event like "ObserveTimeSteps".
      input_file: The input file read in as a dictionary.

    Returns: The event as a dictionary, or None if the event wasn't found.
    """
    for trigger_and_events in input_file["EventsAndTriggers"]:
        for event in trigger_and_events["Events"]:
            if event_name in event:
                return event[event_name]
    return None
