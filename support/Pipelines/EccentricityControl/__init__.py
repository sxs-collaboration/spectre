# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from .EccentricityControl import eccentricity_control_command

if __name__ == "__main__":
    eccentricity_control_command(help_option_names=["-h", "--help"])
