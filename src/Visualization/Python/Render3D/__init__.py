# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from spectre.support.CliExceptions import RequiredChoiceError


class Render3DCommands(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "bbh",
            "clip",
            "domain",
        ]

    def get_command(self, ctx, name):
        if name == "clip":
            from spectre.Visualization.Render3D.Clip import render_clip_command

            return render_clip_command
        elif name == "domain":
            from spectre.Visualization.Render3D.Domain import (
                render_domain_command,
            )

            return render_domain_command
        elif name == "bbh":
            from spectre.Visualization.Render3D.Bbh import render_bbh_command

            return render_bbh_command
        raise RequiredChoiceError(
            f"The command '{name}' is not implemented.",
            choices=self.list_commands(ctx),
        )


@click.group(cls=Render3DCommands)
def render_3d_command():
    """Renders a 3D visualization of simulation data.

    See subcommands for possible renderings.
    """


if __name__ == "__main__":
    render_3d_command(help_option_names=["-h", "--help"])
