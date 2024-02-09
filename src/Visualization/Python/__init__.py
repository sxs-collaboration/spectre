# Distributed under the MIT License.
# See LICENSE.txt for details.

import click


class PlotCommands(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "along-line",
            "dat",
            "power-monitors",
            "size-control",
        ]

    def get_command(self, ctx, name):
        if name == "along-line":
            from spectre.Visualization.PlotAlongLine import (
                plot_along_line_command,
            )

            return plot_along_line_command
        elif name == "dat":
            from spectre.Visualization.PlotDatFile import plot_dat_command

            return plot_dat_command
        elif name == "power-monitors":
            from spectre.Visualization.PlotPowerMonitors import (
                plot_power_monitors_command,
            )

            return plot_power_monitors_command
        elif name in ["size", "size-control"]:
            from spectre.Visualization.PlotSizeControl import (
                plot_size_control_command,
            )

            return plot_size_control_command

        available_commands = " " + "\n ".join(self.list_commands(ctx))
        raise click.UsageError(
            f"The command '{name}' is not implemented. "
            f"Available commands are:\n{available_commands}"
        )


@click.group(cls=PlotCommands)
def plot_command():
    """Plot data from simulations

    See subcommands for available plots.
    """
    pass
