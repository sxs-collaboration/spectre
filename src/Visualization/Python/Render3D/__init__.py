# Distributed under the MIT License.
# See LICENSE.txt for details.

import click


class Render3DCommands(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "clip",
        ]

    def get_command(self, ctx, name):
        if name == "clip":
            from spectre.Visualization.Render3D.Clip import render_clip_command

            return render_clip_command

        available_commands = " " + "\n ".join(self.list_commands(ctx))
        raise click.UsageError(
            f"The command '{name}' is not implemented. "
            f"Available commands are:\n{available_commands}"
        )


@click.group(cls=Render3DCommands)
def render_3d_command():
    """Renders a 3D visualization of simulation data.

    See subcommands for possible renderings.
    """


if __name__ == "__main__":
    render_3d_command(help_option_names=["-h", "--help"])
