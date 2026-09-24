# Jack's Objective: Modify the script to include an option that allows it to read only the surface horizon data and ignore the volume data if prompted. I.e., if the user specifies where the volume data is, you can just use it. If the user doesn't specify it, ignore and utilize only the surface data. Utilizing an if-else statement, set an option for a default value of None and only include the horizons.

# New Objective: Smooth out horizon surfaces

# Provide an option that allows the user to choose color based on the ricci scalar found inside the bbh file (solid white if not specified).

# Make a movie that allows the BH's to come together and allow the individual horizons to disappear.

# Make a unit test - talk to Alex and Lovelace

#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import faulthandler

# force prints to show up immediately and dump C stacks on hard crashes
import os
import sys

faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)
os.environ["PARAVIEW_DEBUG"] = "1"

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


from vtkmodules.vtkCommonCore import vtkObject

vtkObject.GlobalWarningDisplayOff()


def ah_vis(ah_xmf, render_view, volume_src=None, use_ricci=False):
    import paraview.simple as pv

    # 1) Read & translate
    reader = pv.XDMFReader(registrationName="Reader", FileNames=[ah_xmf])
    pv.UpdatePipeline(proxy=reader)
    print("Passed Reader")

    trans = pv.Transform(registrationName="Transform", Input=reader)
    trans.Transform.Translate = [0, 0, 2]
    pv.UpdatePipeline(proxy=trans)
    print("Passed Transform")

    # 2) Extract pure polydata
    ext = pv.ExtractSurface(registrationName="ExtractSurface", Input=trans)
    pv.UpdatePipeline(proxy=ext)
    print("Passed Extract")

    # 2.5) Merge blocks so we get a single vtkPolyData
    merged = pv.MergeBlocks(registrationName="MergeBlocks", Input=ext)
    pv.UpdatePipeline(proxy=merged)
    print("Passed Merge")

    # 2.6) Extract a pure vtkPolyData surface
    surf = pv.ExtractSurface(registrationName="Flatten", Input=merged)
    pv.UpdatePipeline(proxy=surf)
    print("Passed Surface")

    # 3) (Optional) Triangulate if you really need triangles
    tri = pv.Triangulate(registrationName="Triangulate", Input=surf)
    pv.UpdatePipeline(proxy=tri)
    print(
        "Cells after Triangulate:",
        tri.GetClientSideObject().GetOutputDataObject(0).GetNumberOfCells(),
    )
    print("Passed Triangulate")

    # 4) Subdivide
    subdiv = pv.LoopSubdivision(registrationName="Subdivide", Input=tri)
    subdiv.NumberofSubdivisions = 1
    pv.UpdatePipeline(proxy=subdiv)
    print("Passed Subdivide")

    # 5) Smooth
    smooth = pv.Smooth(registrationName="SmoothHorizon", Input=subdiv)
    smooth.NumberofIterations = 200
    smooth.Convergence = 0.01
    pv.UpdatePipeline(proxy=smooth)
    print("Passed Smooth")

    # 6) Show only the smoothed mesh
    rep = pv.Show(smooth, render_view, "UnstructuredGridRepresentation")
    rep.InterpolateScalarsBeforeMapping = True
    rep.Representation = "Surface"

    # Color by the user‐selected scalar
    if use_ricci:
        if volume_src is not None:
            # resample volume’s SpatialRicciScalar onto the smooth horizon
            samp = pv.ResampleWithDataset(
                Source=volume_src, DestinationMesh=smooth
            )
            pv.UpdatePipeline(samp)
            disp = pv.Show(samp, render_view, "UnstructuredGridRepresentation")
            # … color by SpatialRicciScalar …
        elif "RicciScalar" in reader.PointData.keys():
            # horizon file itself has RicciScalar – color that directly
            lut = pv.GetColorTransferFunction("RicciScalar")
            lut.ApplyPreset("Viridis (matplotlib)", True)
            pv.ColorBy(rep, ("POINTS", "RicciScalar"))
            rep.LookupTable = lut
        else:
            # horizon file doesn’t even have RicciScalar → fall back
            pv.ColorBy(rep, None)
    else:
        # fallback: solid white
        disp = rep
        disp.AmbientColor = [1, 1, 1]
        disp.DiffuseColor = [1, 1, 1]
        pv.ColorBy(disp, None)

    # 7) Hide upstream
    for src in (reader, trans, ext, merged, tri, subdiv):
        pv.Hide(src)

    render_view.Update()
    print("▶ smoothed horizon:", ah_xmf)


def render_bbh(
    output: str,
    volume_xmf: str = None,  # now defaults to None
    aha_xmf: str = None,  # now defaults to None
    ahb_xmf: str = None,  # now defaults to None
    time_step: int = 0,
    color_ricci: bool = False,
    animate: bool = False,
    camera_angle: str = "Side",
    zoom_factor: float = 1.0,
    color_map: str = "Rainbow Uniform",
    show_grid: bool = False,
    show_time: bool = False,
):
    """
    Generate Pictures from XMF files for BBH Visualizations

    Generates pictures from BBH runs using the XMF files generated using
    generate-xdmf. This script requires that the Lapse and RicciScalar
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

    To splice all the pictures into a video, try using FFmpeg
    """

    import paraview.simple as pv

    # Surface-only mode: no volume data supplied
    if volume_xmf is None:
        render_view = pv.GetActiveViewOrCreate("RenderView")
        # overlay A/B horizons and then save directly
        if aha_xmf:
            ah_vis(aha_xmf, render_view, use_ricci=color_ricci)
        if ahb_xmf:
            ah_vis(ahb_xmf, render_view, use_ricci=color_ricci)
        # set up camera exactly as in the full routine:
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
        camera = pv.GetActiveCamera()
        pv.ResetCamera()
        camera.Zoom(zoom_factor)
        # and finally write out a screenshot or animation:
        """
        if animate:
            pv.animation(output, render_view)
        else:
            pv.Render()
            pv.SaveScreenshot(output, render_view)
        return
        """
        if animate:
            anim = pv.GetAnimationScene()
            anim.PlayMode       = "Sequence"
            # reuse the same reader you passed into ah_vis:
            horizon_reader = pv.XDMFReader(FileNames=[aha_xmf])
            anim.StartTime      = horizon_reader.TimestepValues[0]
            anim.EndTime        = horizon_reader.TimestepValues[-1]
            anim.NumberOfFrames = 120

            pv.SaveAnimation(
                output,
                render_view,
                FrameRate=30,
                ImageResolution=[1920, 1080],
            )
        else:
            pv.Render()
            pv.SaveScreenshot(output, render_view)
        return
        # Skip the full pipeline and render only horizons if no volume_xmf.
        # Otherwise run the normal slice/warp/color steps below.

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

    # Check for Lapse and RicciScalar
    variables = list(volume_files_xmf.PointData.keys())
    assert (
        "Lapse" in variables
    ), "Lapse not found in volume data, the script will not work correctly."
    assert "RicciScalar" in variables, (
        "RicciScalar not found in volume data, the script will not work"
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
    warp_by_scalar.Scalars = ["POINTS", "RicciScalar"]
    warp_by_scalar.ScaleFactor = 2.5
    warp_by_scalar.Normal = [0.0, 0.0, -1.0]
    warp_by_scalar_display = pv.Show(
        warp_by_scalar, render_view, "GeometryRepresentation"
    )
    warp_by_scalar_display.SetScalarBarVisibility(render_view, False)

    # Apparent Horizon Visualization
    if aha_xmf:
        ah_vis(
            aha_xmf,
            render_view,
            volume_src=volume_files_xmf,
            use_ricci=color_ricci,
        )
    if ahb_xmf:
        ah_vis(
            ahb_xmf,
            render_view,
            volume_src=volume_files_xmf,
            use_ricci=color_ricci,
        )

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
    camera = pv.GetActiveCamera()
    pv.ResetCamera()
    camera.Zoom(zoom_factor)

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
@click.option(
    "--volume-xmf",
    "-v",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=False,
    default=None,
    help=(
        "Optional XMF file for the volume data. If omitted, only horizons are"
        " drawn."
    ),
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
@click.option("zoom_factor", "--zoom", help="Zoom factor.", default=1.0)
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
@click.option(
    "--color-ricci",
    is_flag=True,
    help=(
        "…If given (and volume-XMF is provided),"
        "resample RicciScalar"
        "onto each horizon and color by it."
    ),
)
def render_bbh_command(**kwargs):
    _rich_traceback_guard = True
    render_bbh(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_bbh_command(help_option_names=["-h", "--help"])
