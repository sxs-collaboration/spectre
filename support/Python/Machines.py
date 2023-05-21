# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Support for host machines, such as supercomputers.

Machines are defined as YAML files in 'support/Machines/'. To add support for a
new machine, add a YAML file that defines a `Machine:` key with the attributes
listed in the `Machine` class below.

To select a machine, specify the `MACHINE` option when configuring the CMake
build.
"""

import os
from dataclasses import dataclass

# functools.cache was added in Py 3.9. Fall back to 'lru_cache' in earlier
# versions, which is pretty much the same but slightly slower.
try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

import yaml


@dataclass(frozen=True)
class Machine(yaml.YAMLObject):
    """A machine we know how to run on, such as a particular supercomputer.

    Many configuration options for job submission are hardcoded in the submit
    script for the machine (such as the total number of available cores per
    node). Here we provide parameters that we want to adjust when scheduling
    jobs, but that have sensible defaults based on our experience running on the
    machine. Parameters are set to 'None' when they have no sensible default for
    the machine.

    Attributes:
      Name: A short name for the machine. Must match the YAML file name.
      Description: A description of the machine. Give some basic context and
        any information that may help people get started using the machine.
        Provide links to wiki pages, signup pages, etc., for additional
        information.
      DefaultProcsPerNode: Default number of worker threads spawned per node.
        It is often advised to leave one core per node or socket free for
        communication, so this might be the number of cores or hyperthreads
        per node minus one.
      DefaultQueue: Default queue that jobs are submitted to. On Slurm systems
        you can see the available queues with `sinfo`.
      DefaultTimeLimit: Default wall time limit for submitted jobs. For
        acceptable formats, see: https://slurm.schedmd.com/sbatch.html#OPT_time
    """

    yaml_tag = "!Machine"
    yaml_loader = yaml.SafeLoader
    # The YAML machine files can have these attributes:
    Name: str
    Description: str
    DefaultProcsPerNode: int
    DefaultQueue: str
    DefaultTimeLimit: str


# Parse YAML machine files as Machine objects
yaml.SafeLoader.add_path_resolver("!Machine", ["Machine"], dict)


class UnknownMachineError(Exception):
    """Indicates we were unsuccessful in identifying the current machine"""

    pass


@cache
def this_machine(
    machinefile_path=os.path.join(os.path.dirname(__file__), "Machine.yaml")
) -> Machine:
    """Determine the machine we are running on.

    Raises `UnknownMachineError` if no machine was selected. Specify the
    `MACHINE` option in the CMake build configuration to select a machine, or
    pass the `machinefile_path` argument to this function.

    Arguments:
      machinefile_path: Path to a YAML file that describes the current machine.
        Defaults to the machine selected in the CMake build configuration.
    """
    if not os.path.exists(machinefile_path):
        raise UnknownMachineError(
            "No machine was selected. Specify the 'MACHINE' option when "
            "configuring the build with CMake. If you are running on a new "
            "machine, please add it to 'support/Machines/'. The machine file "
            f"was expected at the following path:\n  {machinefile_path}"
        )
    with open(machinefile_path, "r") as open_machinefile:
        return yaml.safe_load(open_machinefile)["Machine"]
