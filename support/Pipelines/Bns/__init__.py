# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from spectre.support.CliExceptions import RequiredChoiceError


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class Bns(click.Group):
    def list_commands(self, ctx):
        return [
            "compute-trajectories",
        ]

    def get_command(self, ctx, name):
        if name == "compute-trajectories":
            from .ComputeTrajectories import compute_trajectories_command

            return compute_trajectories_command


@click.group(name="bns", cls=Bns)
def bns_pipeline():
    """Pipeline for binary neutron star simulations."""
    pass


if __name__ == "__main__":
    bns_pipeline(help_option_names=["-h", "--help"])
