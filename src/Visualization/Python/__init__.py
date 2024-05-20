# Distributed under the MIT License.
# See LICENSE.txt for details.

import click


class PlotCommands(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "along-line",
            "cce",
            "control-system",
            "dat",
            "elliptic-convergence",
            "memory-monitors",
            "power-monitors",
            "size-control",
            "slice",
        ]

    def get_command(self, ctx, name):
        if name == "along-line":
            from spectre.Visualization.PlotAlongLine import (
                plot_along_line_command,
            )

            return plot_along_line_command
        if name == "cce":
            from spectre.Visualization.PlotCce import plot_cce_command

            return plot_cce_command
        elif name == "control-system":
            from spectre.Visualization.PlotControlSystem import (
                plot_control_system_command,
            )

            return plot_control_system_command
        elif name == "dat":
            from spectre.Visualization.PlotDatFile import plot_dat_command

            return plot_dat_command
        elif name == "elliptic-convergence":
            from spectre.Visualization.PlotEllipticConvergence import (
                plot_elliptic_convergence_command,
            )

            return plot_elliptic_convergence_command
        elif name == "memory-monitors":
            from spectre.Visualization.PlotMemoryMonitors import (
                plot_memory_monitors_command,
            )

            return plot_memory_monitors_command
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
        elif name == "slice":
            from spectre.Visualization.PlotSlice import plot_slice_command

            return plot_slice_command

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
