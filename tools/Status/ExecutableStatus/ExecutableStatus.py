# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from spectre.Visualization.ReadH5 import to_dataframe
from spectre.Visualization.ReadInputFile import find_event

logger = logging.getLogger(__name__)


class ExecutableStatus:
    """Base class for providing status information for SpECTRE executables.

    Subclasses read output from executables and provide status information.
    For example, a subclass for a binary black hole evolution executable can
    provide the current separation and number of orbits.

    Properties:
      executable_name_patterns: List of regex patterns that match the executable
        names the subclass applies to. For example, r"^Evolve" matches all
        executables that start with "Evolve".
      fields: Dictionary of all field names the subclass provides and their
        units. For example: {"Time": "M", "Orbits": None} if the subclass
        provides time in units of mass and orbits with no unit.
    """
    executable_name_patterns: List[str] = []
    fields: Dict[str, Optional[str]] = {}

    def status(self, input_file: dict, work_dir: str) -> dict:
        """Provide status information of an executable run.

        Arguments:
          input_file: The input file read in as a dictionary.
          work_dir: The working directory of the executable run. Paths in the
            input file are relative to this directory.

        Returns: Dictionary of 'fields' and their values. For example:
          {"Time": 10.5, "Orbits": 3}. Not all fields must be provided if they
          can't be determined.

        Note: Avoid raising exceptions in this function. It is better to provide
          only a subset of fields than to raise an exception. For example, if
          there's an error reading the number of orbits, intercept and log the
          exception and just skip providing the number of orbits.
        """
        return {}

    def format(self, field: str, value: Any) -> str:
        """Transform status values to human-readable strings.

        Arguments:
          field: The name of the field, as listed in 'fields' (e.g. "Time").
          value: The status value of the field, as returned by 'status()'.

        Returns: Human-readable representation of the value.
        """
        raise NotImplementedError


class EvolutionStatus(ExecutableStatus):
    """An 'ExecutableStatus' subclass that matches all evolution executables.

    This is a fallback if no more specialized subclass is implemented. It just
    determines the current time and run speed. This class can be subclassed
    further to use the 'time_status' function in subclasses.
    """
    executable_name_patterns = [r"^Evolve"]
    fields = {
        "Time": None,
        "Speed": "1/h",
    }

    def time_status(self, input_file: dict, open_reductions_file) -> dict:
        """Report the simulation time and speed.

        Uses the 'ObserveTimeStep' event, so the status information is only as
        current as the output frequency of the event. The reported time is the
        most recent output of the 'ObserveTimeStep' event, and the reported
        speed is determined from the two most recent outputs. Since the walltime
        at which different elements in the computational domain reach the
        timestep varies because of the lack of a global synchronization point in
        the task-based parallel evolution, this function returns the average
        between the minimum and maximum walltime.

        Arguments:
          input_file: The input file read in as a dictionary.
          open_reductions_file: The open h5py reductions data file.

        Returns: Status fields "Time" in code units and "Speed" in simulation
          time per hour.
        """
        observe_time_event = find_event("ObserveTimeStep", input_file)
        if observe_time_event is None:
            return {}
        subfile_name = observe_time_event["SubfileName"] + ".dat"
        logger.debug(f"Reading time steps from subfile: '{subfile_name}'")
        try:
            time_steps = to_dataframe(open_reductions_file[subfile_name])
        except:
            logger.debug("Unable to read time steps.", exc_info=True)
            return {}
        result = {
            "Time": time_steps.iloc[-1]["Time"],
        }
        # Estimate simulation speed
        try:
            dt_sim = time_steps.iloc[-1]["Time"] - time_steps.iloc[-2]["Time"]
            dt_wall_min = (time_steps.iloc[-1]["Minimum Walltime"] -
                           time_steps.iloc[-2]["Minimum Walltime"])
            dt_wall_max = (time_steps.iloc[-1]["Maximum Walltime"] -
                           time_steps.iloc[-2]["Maximum Walltime"])
            speed = (dt_sim / dt_wall_min +
                     dt_sim / dt_wall_max) / 2. * 60. * 60.
            result["Speed"] = speed
        except:
            logger.debug("Unable to estimate simulation speed.", exc_info=True)
        return result

    def status(self, input_file, work_dir):
        try:
            reductions_file = input_file["Observers"]["ReductionFileName"]
            open_reductions_file = h5py.File(
                os.path.join(work_dir, reductions_file + ".h5"), "r")
        except:
            logger.debug("Unable to open reductions file.", exc_info=True)
            return {}
        with open_reductions_file:
            return self.time_status(input_file, open_reductions_file)

    def format(self, field, value):
        if field in ["Time", "Speed"]:
            return f"{value:g}"
        raise ValueError
