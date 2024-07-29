#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich.columns

logger = logging.getLogger(__name__)


def _parse_step(ctx, param, value):
    if value is None:
        return None
    if value.lower() == "first":
        return 0
    if value.lower() == "last":
        return -1
    return int(value)


def ah_vis(ah_xmf: str, render_view: str):
    """Helper function for visualizing the apparent horizons of the objects.

    Arguments:
    ah_xmf: Path to the xmf file of the object.
    render_view: The current view in paraview to add the horizon to."""
    import paraview.simple as pv

    Ah_xmf = pv.XDMFReader(registrationName=ah_xmf, FileNames=[ah_xmf])
    transform_1 = pv.Transform(registrationName="Transform1", Input=Ah_xmf)
    transform_1.Transform = "Transform"
    transform_1.Transform.Translate = [0.0, 0.0, 2.0]
    transform_1_display = pv.Show(
        transform_1, render_view, "UnstructuredGridRepresentation"
    )
    transform_1_display.SetScalarBarVisibility(render_view, False)
    # Sets apparent horizon color to black
    transform_1_display.AmbientColor = [0.0, 0.0, 0.0]
    transform_1_display.DiffuseColor = [0.0, 0.0, 0.0]

    render_view.Update()
    pv.ColorBy(transform_1_display, None)


def render_bbh(
    volume_xmf: str,
    output: str,
    aha_xmf: str,
    ahb_xmf: str,
    time_step: int = 0,
    animate: bool = False,
    camera_angle: str = "Side",
    color_map: str = "Rainbow Uniform",
    show_grid: bool = False,
    show_time: bool = False,
):
    """Generate Pictures from XMF files for BBH Visualizations

    Generates pictures from BBH runs using the XMF files generated using
    generate-xdmf. This script requires that the Lapse and SpatialRicciScalar
    were output in the volume data.

    Arguments:

      volume_xmf: Path to the volume data xmf file.
      output: Name of output file generated from paraview. Include extensions
      such as '.png'
      aha_xmf: Path to the apparent horizon xmf file for object A.
      ahb_xmf: Path to the apparent horizon xmf file for object B.
      camera_angle: Specified camera angle, defaults to Side if empty. Other
      possible angles Top and Wide
      color_map: Color map for the lapse, defaults to 'Rainbow Uniform'. Other
      color maps include 'Inferno (matplotlib)', 'Viridis (matplotlib)', etc.
      show_grid: Shows the grid lines of the domain.
      show_time: Shows the simulation time.

    To splice all the pictures into a video, try using FFmpeg"""
    import paraview.simple as pv

    version = pv.GetParaViewVersion()
    if version < (5, 11) or version > (5, 11):
        logger.warning(
            "WARNING: Your Paraview version is not 5.11, "
            "the script may not work correctly."
        )

    # Volume Data Visualization
    volume_files_xmf = pv.XDMFReader(
        registrationName=volume_xmf, FileNames=[volume_xmf]
    )

    # Check for Lapse and SpatialRicciScalar
    variables = list(volume_files_xmf.PointData.keys())
    assert (
        "Lapse" in variables
    ), "Lapse not found in volume data, the script will not work correctly."
    assert "SpatialRicciScalar" in variables, (
        "SpatialRicciScalar not found in volume data, the script will not work"
        " correctly."
    )

    render_view = pv.GetActiveViewOrCreate("RenderView")

    # Color the grid
    color_transfer_function = pv.GetColorTransferFunction("Lapse")
    color_transfer_function.Discretize = 0
    color_transfer_function.ApplyPreset(color_map, True)
    color_transfer_function.InvertTransferFunction()

    # Slice volume data
    slice = pv.Slice(registrationName="slice", Input=volume_files_xmf)
    slice.SliceType = "Plane"
    slice.HyperTreeGridSlicer = "Plane"
    slice.SliceOffsetValues = [0.0]
    slice.SliceType.Normal = [0.0, 0.0, 1.0]
    slice.Triangulatetheslice = 0

    # Warp grid by spatial ricci scalar
    warp_by_scalar = pv.WarpByScalar(
        registrationName="WarpByScalar", Input=slice
    )
    warp_by_scalar.Scalars = ["POINTS", "SpatialRicciScalar"]
    warp_by_scalar.ScaleFactor = 2.5
    warp_by_scalar.Normal = [0.0, 0.0, -1.0]
    warp_by_scalar_display = pv.Show(
        warp_by_scalar, render_view, "GeometryRepresentation"
    )
    warp_by_scalar_display.SetScalarBarVisibility(render_view, False)

    # Apparent Horizon Visualization
    if aha_xmf:
        ah_vis(aha_xmf, render_view)
    if ahb_xmf:
        ah_vis(ahb_xmf, render_view)

    if show_grid:
        warp_by_scalar_display.Representation = "Surface With Edges"

    pv.LoadPalette(paletteName="GradientBackground")
    render_view.OrientationAxesVisibility = 0
    pv.SetActiveSource(warp_by_scalar)
    warp_by_scalar_display.Opacity = 0.8
    pv.ColorBy(warp_by_scalar_display, ("POINTS", "Lapse"))
    layout = pv.GetLayout()
    layout.SetSize(1920, 1080)

    # Camera placements
    # Top down view
    if camera_angle == "Top":
        render_view.CameraPosition = [0.0, 0.0, 36.90869716569761]
        render_view.CameraFocalPoint = [0.0, 0.0, 0.6894899550131899]
        render_view.CameraViewUp = [0, 1, 0]
        render_view.CameraParallelScale = 424.27024700303446
    # Wide/Inbetween View
    elif camera_angle == "Wide":
        render_view.CameraPosition = [
            -89.0,
            -17.0,
            25.0,
        ]
        render_view.CameraFocalPoint = [
            -0.3921962951264054,
            1.6346750682876983,
            -0.34522248814953405,
        ]
        render_view.CameraViewUp = [
            0.0,
            0.0,
            1.0,
        ]
    # Side View
    else:
        render_view.CameraPosition = [
            -29.944619336722987,
            -3.666072157343372,
            2.895224044348878,
        ]
        render_view.CameraFocalPoint = [
            -0.13267040638072278,
            0.6356115665206243,
            -0.37352608789235847,
        ]
        render_view.CameraViewUp = [0.0, 0.0, 1.0]
        render_view.CameraParallelScale = 519.6152422706632

    # Simulation time
    if show_time:
        time_filter = pv.AnnotateTimeFilter(
            registrationName="annotate_time_filter", Input=slice
        )
        time_filter.Format = "Time: {time:0.2f}M"
        annotate_time_filter_display = pv.Show(
            time_filter, render_view, "TextSourceRepresentation"
        )
        annotate_time_filter_display.FontSize = 45

    # Capture all frames
    animation_scene = pv.GetAnimationScene()
    animation_scene.PlayMode = "Snap To TimeSteps"

    # Save animation/screenshot
    if animate:
        pv.SaveAnimation(output, render_view)
    else:
        render_view.ViewTime = volume_files_xmf.TimestepValues[time_step]
        pv.Render()
        pv.SaveScreenshot(output, render_view)


@click.command(name="bbh", help=render_bbh.__doc__)
@click.argument(
    "volume_xmf",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True
    ),
    required=True,
    help="Output file. Include extension such as '.png'.",
)
@click.option(
    "--aha-xmf",
    "-a",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Optional xmf file for AhA visualization",
)
@click.option(
    "--ahb-xmf",
    "-b",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Optional xmf file for AhB visualization",
)
@click.option(
    "--time-step",
    "-t",
    callback=_parse_step,
    default="first",
    show_default=True,
    help=(
        "Select a time step. Specify '-1' or 'last' to select the last time"
        " step."
    ),
)
@click.option(
    "--animate", is_flag=True, help="Produce an animation of all time steps."
)
@click.option(
    "--camera-angle",
    "-c",
    default="Side",
    type=click.Choice(["Side", "Top", "Wide"]),
    help=(
        "Determines which camera angle to use: Default is the Side view.Top"
        " view is right above the excisions at t = 0. Wide is further out and"
        " inbetween Side and Top view"
    ),
)
@click.option(
    "--color-map",
    "-m",
    default="Rainbow Uniform",
    help=(
        'Determines how to color the domain, common color maps are "Inferno'
        ' (matplotlib)", "Viridis (matplotlib). Defaults to Rainbow Uniform."'
    ),
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Show grid lines",
)
@click.option(
    "--show-time",
    is_flag=True,
    help="Show simulation time",
)
def render_bbh_command(**kwargs):
    _rich_traceback_guard = True
    render_bbh(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_bbh_command(help_option_names=["-h", "--help"])
