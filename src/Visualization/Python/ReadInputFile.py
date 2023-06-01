# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Tools for parsing YAML input files

To simply read an input file as a dictionary, use a standard YAML parser like
PyYAML or ruaml.yaml::

    import yaml
    with open(input_file_path) as open_input_file:
        metadata, input_file = yaml.safe_load_all(open_input_file)

It's also possible to load just the metadata without the overhead of parsing the
full input file::

    with open(input_file_path) as open_input_file:
        metadata = next(yaml.safe_load_all(open_input_file))

The functions in this module provide additional functionality to work with input
files.
"""


def find_event(event_name: str, input_file: dict) -> dict:
    """Find a particular event in the "EventsAndTriggers" of the input file.

    Arguments:
      event_name: The name of an event like "ObserveTimeSteps".
      input_file: The input file read in as a dictionary.

    Returns: The event as a dictionary, or None if the event wasn't found.
    """
    for trigger_and_events in input_file["EventsAndTriggers"]:
        try:
            for event in trigger_and_events["Events"]:
                if event_name in event:
                    return event[event_name]
        except TypeError:
            # Backwards compatibility for input files without metadata (can be
            # removed once people have rebased)
            for event in trigger_and_events[1]:
                if event_name in event:
                    return event[event_name]
    return None


def find_phase_change(phase_change_name: str, input_file: dict) -> dict:
    """Find a particular phase change in the "PhaseChangeAndTriggers"

    Arguments:
      phase_change_name: The name of a phase change like
        "CheckpointAndExitAfterWallclock".
      input_file: The input file read in as a dictionary.

    Returns: The phase change as a dictionary, or None if the phase change
      wasn't found.
    """
    if not "PhaseChangeAndTriggers" in input_file:
        return None
    for trigger_and_phase_changes in input_file["PhaseChangeAndTriggers"]:
        for phase_change in trigger_and_phase_changes["PhaseChanges"]:
            if isinstance(phase_change, str):
                if phase_change == phase_change_name:
                    return {}
            elif phase_change_name in phase_change:
                return phase_change[phase_change_name]
    return None
