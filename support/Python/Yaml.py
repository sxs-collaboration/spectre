# Distributed under the MIT License.
# See LICENSE.txt for details.

from pathlib import Path

import numpy as np
import yaml


# Write `pathlib.Path` objects to YAML as plain strings
def _path_representer(dumper: yaml.Dumper, path: Path) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(path))


# Write `numpy.float64` as regular floats
def _numpy_representer(
    dumper: yaml.Dumper, value: np.float64
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(value))


SafeDumper = yaml.SafeDumper
SafeDumper.add_multi_representer(Path, _path_representer)
SafeDumper.add_multi_representer(np.float64, _numpy_representer)
