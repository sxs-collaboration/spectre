#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import glob
import logging
import os
import paraview.simple
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def ah_vis(ah_xmf, render_view):
    Ah_xmf = XDMFReader(registrationName=ah_xmf, FileNames=[ah_xmf])
    Ah_xmf.PointArrayStatus = ['RicciScalar']
    ricciScalarLUT = GetColorTransferFunction('RicciScalar')
    ricciScalarPWF = GetOpacityTransferFunction('RicciScalar')
    Ah_xmf_Display = Show(Ah_xmf, render_view,
                          'UnstructuredGridRepresentation')
    Ah_xmf_Display.SetScalarBarVisibility(render_view, False)
    render_view.Update()
    transform1 = Transform(registrationName='Transform1', Input=Ah_xmf)
    transform1.Transform = 'Transform'
    transform1.Transform.Translate = [0.0, 0.0, 2.5]
    transform1_display = Show(transform1, render_view,
                              'UnstructuredGridRepresentation')
    Hide(Ah_xmf, render_view)
    transform1_display.SetScalarBarVisibility(render_view, False)
    render_view.Update()
    ColorBy(transform1_display, None)
    HideScalarBarIfNotNeeded(ricciScalarLUT, render_view)
    transform1_display.AmbientColor = [0.0, 0.0, 0.0]
    transform1_display.DiffuseColor = [0.0, 0.0, 0.0]
    Hide3DWidgets(proxy=transform1.Transform)


def render_bbh(output_dir, volume_xmf, aha_xmf, ahb_xmf, camera_angle):
    """Generate Pictures from XMF files for BBH Visualizations

    Generate pictures from BBH runs using the XMF files generated using
    generate-xdmf. For files to be read properly, the XMF files being pointed to
    should be in the same directory as the Volume.h5 and Surfaces.h5 files.

    To splice all the pictures into a video, look into FFmpeg"""
    version = paraview.simple.GetParaViewVersion()
    if (version < (5, 10) or version > (5, 10)):
        print("WARNING: Your Paraview version is not 5.10, "
              "the script may not work correctly.")

    # Volume Data Visualization
    volume_files_xmf = XDMFReader(registrationName=volume_xmf,
                                  FileNames=[volume_xmf])
    volume_files_xmf.PointArrayStatus = ['Lapse', 'SpatialRicciScalar']
    animation_scene1 = GetAnimationScene()
    animation_scene1.UpdateAnimationUsingDataTimeSteps()
    render_view1 = GetActiveViewOrCreate('RenderView')
    lapse_LUT = GetColorTransferFunction('Lapse')
    lapse_PWF = GetOpacityTransferFunction('Lapse')
    render_view1.ResetCamera(False)
    materialLibrary1 = GetMaterialLibrary()
    render_view1.Update()
    slice1 = Slice(registrationName='Slice1', Input=volume_files_xmf)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.SliceOffsetValues = [0.0]
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]
    Hide(volume_files_xmf, render_view1)
    render_view1.Update()
    lapse_LUT.Discretize = 0
    slice1.Triangulatetheslice = 0
    render_view1.Update()
    warp_by_scalar1 = WarpByScalar(registrationName='WarpByScalar1',
                                   Input=slice1)
    warp_by_scalar1.Scalars = ['POINTS', 'Lapse']
    warp_by_scalar1.Scalars = ['POINTS', 'SpatialRicciScalar']
    warp_by_scalar1.ScaleFactor = 2.5
    warp_by_scalar1.Normal = [0.0, 0.0, -1.0]
    warp_by_scalar1_display = Show(warp_by_scalar1, render_view1,
                                   'GeometryRepresentation')
    Hide(slice1, render_view1)
    warp_by_scalar1_display.SetScalarBarVisibility(render_view1, False)
    render_view1.Update()
    lapse_LUT.ApplyPreset('Rainbow Uniform', True)
    lapse_LUT.InvertTransferFunction()

    # Apparent Horizon Visualization
    if (aha_xmf):
        ah_vis(aha_xmf, render_view1)
    if (ahb_xmf):
        ah_vis(ahb_xmf, render_view1)

    LoadPalette(paletteName='GradientBackground')
    render_view1.OrientationAxesVisibility = 0
    SetActiveSource(warp_by_scalar1)
    warp_by_scalar1_display.Opacity = 0.8
    layout1 = GetLayout()
    layout1.SetSize(1920, 1080)

    # Camera placements for views
    # Top down view
    if (camera_angle == 1):
        render_view1.CameraPosition = [0.0, 0.0, 36.90869716569761]
        render_view1.CameraFocalPoint = [0.0, 0.0, 0.6894899550131899]
        render_view1.CameraViewUp = [0, 1, 0]
        render_view1.CameraParallelScale = 424.27024700303446
    # Wide/Inbetween View
    elif (camera_angle == 2):
        render_view1.CameraPosition = [
            -30.093571816984163, -18.55482256667294, 8.558827016411096
        ]
        render_view1.CameraFocalPoint = [
            6.698318283229103e-17, 4.073218385661855e-17, 0.6894899550131903
        ]
        render_view1.CameraViewUp = [0.0, 0.0, 1.0]
        render_view1.CameraParallelScale = 424.27024700303446
    # Side View
    else:
        render_view1.CameraPosition = [
            -29.944619336722987, -3.666072157343372, 2.895224044348878
        ]
        render_view1.CameraFocalPoint = [
            -0.13267040638072278, 0.6356115665206243, -0.37352608789235847
        ]
        render_view1.CameraViewUp = [0.0, 0.0, 1.0]
        render_view1.CameraParallelScale = 519.6152422706632

    # save animation
    SaveAnimation(os.path.join(output_dir, 'Binary_Pic.png'),
                  render_view1,
                  ImageResolution=[1920, 1080],
                  FrameRate=60)


@click.command(help=render_bbh.__doc__)
@click.option('--output-dir',
              '-o',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              required=True,
              help="Output directory where pictures will be placed")
@click.option('--volume-xmf',
              '-v',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=False,
                              readable=True),
              required=True,
              help="Xmf file for VolumeData")
@click.option(
    '--aha-xmf',
    '-a',
    type=click.Path(exists=True, file_okay=True, dir_okay=False,
                    readable=True),
    help="Optional xmf file for AhA for apparent horizon visualization")
@click.option(
    '--ahb-xmf',
    '-b',
    type=click.Path(exists=True, file_okay=True, dir_okay=False,
                    readable=True),
    help="Optional xmf file for AhB for apparent horizon visualization")
@click.option(
    '--camera-angle',
    '-c',
    default=0,
    type=click.Choice(['0', '1', '2']),
    help="Determines which camera angle to use: Default 0 is a side view,"
    "1 is a top down, 2 is further out but inbetween side and top down view")
def render_bbh_command(**kwargs):
    _rich_traceback_guard = True
    render_bbh(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_bbh_command(help_option_names=["-h", "--help"])
