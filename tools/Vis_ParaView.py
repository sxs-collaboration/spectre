#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from paraview.simple import *
import sys
import yaml
import numpy as np
import math
import argparse
# input file class
from InputFile import InputFile


def parse_cmd_line():
    '''
    parse command-line arguments
    :return: dictionary of the command-line args, dashes are underscores
    '''
    parser = argparse.ArgumentParser(description='Visualization using Paraview',
                                     formatter_class=\
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-file', type=str, required=True,
                        help="provide path to"
                        "yaml file containing visualization parameters")
    parser.add_argument('--save', type=str, required=True,
                        help="set to the name of output file to be written."
                        "For animations this saves a png file at each timestep"
                        "and for stills, a png")
    return vars(parser.parse_args())


def set_default_camera(render_view):
    '''
    Set view and camera properties to a default
    specified below
    '''
    render_view.AxesGrid = 'GridAxes3DActor'
    render_view.CenterOfRotation = [0.0, 0.0, -22.98529815673828]
    render_view.StereoType = 0
    render_view.CameraPosition = [-95.16305892510273,
                                  754.3761479590994, 426.10491067642846]
    render_view.CameraFocalPoint = [
        43.11431452669897, 23.554652286151217, -46.78071794111179]
    render_view.CameraViewUp = [
        0.09790398117889891, -0.5275190119519493, 0.843882990999677]
    render_view.CameraParallelScale = 715.9367631158923
    render_view.Background = [0.0, 0.0, 0.0]
    return render_view


def load_input_file(input_file_name):
    '''
    Read data from yaml input file
    Name/path of yaml file is specified in command line arguments
    '''
    # Check yaml file existence:
    try:
        input_stream = open(input_file_name, 'r')
    except FileNotFoundError:
        sys.exit("Input file was not found at location: " + input_file_name)
    # Try for error in Yaml formatting
    # If an error is found, the mark is returned and printed
    try:
        input_dictionary = yaml.load(input_stream)
    except yaml.YAMLError, exc:
        print("The input file cannot be parsed because of a syntax error.")
        if hasattr(exc, 'problem_mark'):
            mark = exc.problem_mark
            sys.exit("The syntax error was encountered at row %s column %s"
                     % (mark.line + 1, mark.column + 1))
    # Create instance of InputFile class and load data
    input_file = InputFile(input_dictionary)
    return input_file


def read_xdmf(xdmf_file_path):
    '''
    Read data from XDMF file specified in yaml file.
    Set source and view variables.
    '''
    try:
        xdmf_reader = XDMFReader(FileNames=xdmf_file_path)
    except FileNotFoundError:
        sys.exit("No such file at file location specified: " + xdmf_file_path)
    xdmf_reader = GetActiveSource()
    render_view = GetActiveViewOrCreate('RenderView')
    xdmf_reader_display = GetDisplayProperties(xdmf_reader, view=render_view)
    return render_view, xdmf_reader, xdmf_reader_display


def set_representation(representation, render_view, display):
    '''
    Set representation, for example, 'Surface' or 'Wireframe',
    and update view
    '''
    display.Representation = representation
    render_view.Update()
    return render_view, display


def set_color_map(var, render_view, display, color_map):
    '''
    Set the color map to one of preset ParaView color maps.
    '''
    ColorBy(display, ('POINTS', var))
    # ParaView returns a lookup table (LUT):
    variable_lookup_table = GetColorTransferFunction(var)
    variable_lookup_table.ApplyPreset(color_map, True)
    # Show the color bar for the variable being visualized:
    display.SetScalarBarVisibility(render_view, True)
    display.SetScaleArray = ['POINTS', var]
    display.ScaleTransferFunction = 'PiecewiseFunction'
    return display, render_view, variable_lookup_table


def tetrahedralize(scalar_var_source, render_view):
    '''
    Apply tetrahedralize filter and update view
    '''
    tetrahedralize = Tetrahedralize(Input=scalar_var_source)
    Hide(scalar_var_source, render_view)
    scalar_var_display = Show(tetrahedralize, render_view)
    render_view.Update()
    return render_view, scalar_var_display


def set_opacity(var, function_type, opacity_val, render_view,\
                scalar_var_source, var_lookup_table):
    '''
    Set opacity based on option chosen in yaml input file:
    'Constant': some value between 0 and 1
    'Proportional': opacity set proportional to the variable being visualized,
    (Under 'Scalar_variable_properties')
    '''
    # Get range of var array
    var_range = scalar_var_source.PointData.GetArray(var).GetRange()
    # Set variables to construct gaussians
    var_max = var_range[1]
    var_min = var_range[0]

    if function_type == 'Constant':  # Constant Opacity
        var_lookup_table.EnableOpacityMapping = 1
        var_pointwise_function = GetOpacityTransferFunction(var)
        var_pointwise_function.Points = [var_min, opacity_val, 0.5,
                          0.0, var_max, opacity_val, 0.5, 0.0]

    elif function_type == 'Proportional':  # Varying opacity
        var_lookup_table.EnableOpacityMapping = 1
        var_pointwise_function = GetOpacityTransferFunction(var)
        num_points = 200  # Number of points to evaluate opacity function
        num_gauss = 5  # Number of gaussians
        var_values = np.asarray(np.linspace(
            var_min, var_max, num_points))  # Array of var values
        center_values = np.asarray(np.linspace(
            var_min, var_max, num_gauss))  # centers of gaussian
        # amplitudes of gaussians
        amplitude_values = abs(center_values)/max(abs(var_max), abs(var_min))
        sigma = (var_values[1] - var_values[0]) * 2.0

        gaussians = []
        gaussians = [np.asarray([center_values[i], amplitude_values[i]])
                     for i in range(num_gauss)]
        opacity_function = np.zeros(len(var_values))
        for gaussian in gaussians:
            opacity_function += gaussian[1] * np.exp(-1.0 * np.square(
                var_values - gaussian[0])/(2.0 * sigma**2))
        # Create opacity list with var_values, opacity function,
        # midpoint value, sharpness
        # Midpoint value used here is 0.5, sharpness value used is 0.0
        opacity_list = []
        for point in range(num_points):
            opacity_list += [var_values[point],
                             opacity_function[point], 0.5, 0.0]
        var_pointwise_function.Points = opacity_list
    render_view.Update()
    return render_view


def set_vector_color_map(vector_var_display, vector_variable, color_map,
                         render_view):
    '''
    Sets color_map for vector field
    '''
    ColorBy(vector_var_display, ('POINTS', vector_variable, 'Magnitude'))
    var_lookup_table = GetColorTransferFunction(vector_variable)
    var_lookup_table.ApplyPreset(color_map,True)
    vector_var_display.SetScalarBarVisibility(render_view, True)
    vector_var_display.RescaleTransferFunctionToDataRange(True, False)
    render_view.Update()
    return render_view


def apply_clip(clip_properties, var, render_view, var_source):
    '''
    Apply the clip filter based on clip type chosen in Input file
    '''
    clip = Clip(Input=var_source)
    if clip_properties.pv_type == 'Plane':
        clip.ClipType = 'Plane'
        clip.Scalars = ['POINTS', var]
        clip.ClipType.Origin = clip_properties.pv_origin
        clip.ClipType.Normal = clip_properties.pv_normal
    elif clip_properties.pv_type == 'Box':
        clip.ClipType = 'Box'
        clip.Scalars = ['POINTS', var]
        clip.ClipType.Position = clip_properties.pv_position
        clip.ClipType.Rotation = clip_properties.pv_rotation
        clip.ClipType.Scale = clip_properties.pv_scale
    elif clip_properties.pv_type == 'Sphere':
        clip.ClipType = 'Sphere'
        clip.Scalars = ['POINTS', var]
        clip.ClipType.Center = clip_properties.pv_sphere_center
        clip.ClipType.Radius = clip_properties.pv_sphere_radius
    elif clip_properties.pv_type == 'Cylinder':
        clip.ClipType = 'Cylinder'
        clip.Scalars = ['POINTS', var]
        clip.ClipType.Center = clip_properties.pv_cylinder_center
        clip.ClipType.Radius = clip_properties.pv_cylinder_radius
        clip.ClipType.Axis = clip_properties.pv_axis
    # Hide previous data display before filter
    Hide(var_source, render_view)
    display = Show(clip, render_view)
    return render_view, clip, display


def apply_slice(slice_properties, var, render_view, var_source):
    '''
    Apply slice filter based on slice type chosen in Input file
    '''
    slice_pv = Slice(Input=var_source)
    if slice_properties.pv_type == 'Plane':
        slice_pv.SliceType = 'Plane'
        slice_pv.SliceOffsetValues = [0.0]
        slice_pv.SliceType.Origin = slice_properties.pv_origin
        slice_pv.SliceType.Normal = slice_properties.pv_normal
    elif slice_properties.pv_type == 'Box':
        slice_pvSliceType = 'Box'
        slice_pv.SliceOffsetValues = [0.0]
        slice_pv.SliceType.Position = slice_properties.pv_position
        slice_pv.SliceType.Rotation = slice_properties.pv_rotation
        slice_pv.SliceType.Scale = slice_properties.pv_scale
    elif slice_properties.pv_type == 'Sphere':
        slice_pv.SliceType = 'Sphere'
        slice_pv.SliceOffsetValues = [0.0]
        slice_pv.SliceType.Center = slice_properties.pv_sphere_center
        slice_pv.SliceType.Radius = slice_properties.pv_sphere_radius
    elif slice_properties.pv_type == 'Cylinder':
        slice_pv.SliceType = 'Cylinder'
        slice_pv.SliceOffsetValues = [0.0]
        slice_pv.SliceType.Center = slice_properties.pv_cylinder_center
        slice_pv.SliceType.Radius = slice_properties.pv_cylinder_radius
        slice_pv.SliceType.Axis = slice_properties.pv_axis
    # Hide previous data display before filter
    Hide(var_source, render_view)
    display = Show(slice_pv, render_view)
    return render_view, slice_pv, display


def add_vector_field(vector_variable, vector_var_source, render_view):
    '''
    Adds a vector field (glyph) of variable specified in input file.
    '''
    vector_field = Glyph(Input=vector_var_source, GlyphType="Arrow")
    vector_field.OrientationArray = ['POINTS', vector_variable]
    vector_field.ScaleArray = ['POINTS', vector_variable]
    # Set basic properties of the glyph object
    vector_field.ScaleFactor = 40
    vector_field.GlyphType.TipResolution = 6
    vector_field.GlyphType.TipRadius = 0.2
    vector_field.GlyphType.TipLength = 0.5
    vector_field.GlyphType.ShaftResolution = 6
    vector_field.GlyphType.ShaftRadius = 0.08
    vector_field.GlyphMode = 'Uniform Spatial Distribution'
    vector_field.Seed = 10339
    vector_field.MaximumNumberOfSamplePoints = 100
    # Glyph display
    vector_var_display = Show(vector_field, render_view)
    render_view.Update()
    return render_view, vector_var_display, vector_field


def create_neg_scalar_var(scalar_var, var_source):
    '''
    Create a negative scalar array for downward warp
    '''
    calculator = Calculator(Input=var_source)
    neg_scalar_var = 'negative_'+ scalar_var #setting name of new variable array
    calculator.ResultArrayName = neg_scalar_var
    calculator.Function = '-1*abs('+ scalar_var+')'
    return neg_scalar_var, calculator


def create_log_scalar_var(scalar_var, var_source):
    '''
    Create a log scalar array to warp for large ranges of scalar var values
    '''
    calculator = Calculator(Input=var_source)
    neg_log_scalar_var = 'negative_log_'+scalar_var
    calculator.ResultArrayName = neg_log_scalar_var
    calculator.Function = '-1*abs(log10('+scalar_var+'))'
    return neg_log_scalar_var, calculator


def translate_var(warp_by_scalar, var_source, render_view):
    '''
    Translate warp to a point below the
    object in view
    '''
    transform = Transform(Input=warp_by_scalar)
    transform.Transform = 'Transform'
    transform.Transform.Translate = [0.0, 0.0, -15.0]
    Hide(warp_by_scalar, render_view)
    return transform, render_view


def add_scalar_warp(scalar_var, var_source, scale_type, scale_factor,
                    render_view):
    '''
    Create a surface warp.
    Warps by variable being visualized
    '''
    if scale_type == 'Linear':
        scaling_scalar_var, calculator = create_neg_scalar_var(scalar_var,
                                                               var_source)
    elif scale_type == 'Log':
        scaling_scalar_var, calculator = create_log_scalar_var(scalar_var,
                                                               var_source)
    slice_pv = Slice(Input=calculator)
    slice_pv.SliceType = 'Plane'
    slice_pv.SliceOffsetValues = [0.0]
    slice_pv.SliceType.Origin = [0, 0, 0]
    slice_pv.SliceType.Normal = [0.0, 0.0, 1.0]
    # Warp slice
    warp_by_scalar = WarpByScalar(Input=slice_pv)
    warp_by_scalar.Scalars = ['POINTS', scaling_scalar_var]
    warp_by_scalar.ScaleFactor = scale_factor
    warp_by_scalar.UseNormal = 1
    # Translate the warp
    transform, render_view = translate_var(warp_by_scalar, var_source,
                                           render_view)
    # Color translated warp by scalar_var
    transform_display = GetDisplayProperties(transform, render_view)
    ColorBy(transform_display, ('POINTS', scalar_var))
    Show()
    render_view.Update()
    return render_view


def add_vector_warp(vector_var, scalar_var, var_source, scale_type,
                    scale_factor, render_view):
    '''
    Add vector warp to show vector field along with scalar warp
    '''
    # First construct negative var array
    # for downward warp
    if scale_type == 'Linear':
        scaling_scalar_var, calculator = create_neg_scalar_var(scalar_var,
                                                               var_source)
    elif scale_type == 'Log':
        scaling_scalar_var, calculator = create_log_scalar_var(scalar_var,
                                                               var_source)
    slice_pv = Slice(Input=calculator)
    slice_pv.SliceType = 'Plane'
    slice_pv.SliceOffsetValues = [0.0]
    slice_pv.SliceType.Origin = [0, 0, 0]
    slice_pv.SliceType.Normal = [0.0, 0.0, 1.0]
    # Get vector field on slice
    vectors_slice = Glyph(Input=slice_pv, GlyphType='2D Glyph')
    vectors_slice.OrientationArray = ['POINTS', vector_var]
    vectors_slice.ScaleArray = ['POINTS', 'SpatialVelocity']
    vectors_slice.ScaleFactor = scale_factor
    vectors_slice.MaximumNumberOfSamplePoints = 100
    vectors_slice_display = GetDisplayProperties(vectors_slice,view=render_view)
    vectors_slice_display.LineWidth = 3.0
    Hide(vectors_slice, render_view)
    # Warp slice
    warp_by_scalar = WarpByScalar(Input=vectors_slice)
    warp_by_scalar.Scalars = ['POINTS', scaling_scalar_var]
    warp_by_scalar.ScaleFactor = 10
    warp_by_scalar.UseNormal = 1
    # Translate the warp
    transform, render_view = translate_var(warp_by_scalar, var_source,
                                           render_view)
    transform_display = Show(transform, render_view)
    Hide(warp_by_scalar, render_view)
    ColorBy(transform_display, ('POINTS', vector_var, 'Magnitude'))
    render_view.Update()
    return render_view


def save_images(render_view, xdmf_reader, save):
    '''
    Saves single image or multiple images based on number of
    time steps in data. One image is saved per time step.
    '''
    time_steps = xdmf_reader.TimestepValues  # list of timesteps
    render_view.ViewSize = [1920, 1080]

    if type(time_steps) is paraview.servermanager.VectorProperty:
        number_of_time_steps = len(time_steps)
    else:
        number_of_time_steps = 1
        time_steps = [time_steps]

    if number_of_time_steps == 1:
        SaveScreenshot(save + '.png', render_view)
    elif number_of_time_steps > 1:
        anim = GetAnimationScene()
        anim.PlayMode = 'Snap To TimeSteps'
        for time_step_index in range(number_of_time_steps):
            print('Rendering time step', time_step_index)
            anim.AnimationTime = time_steps[time_step_index]
            current_view = GetRenderView()
            #Add time step value to window
            display_time(xdmf_reader, current_view)
            # Generating 6-digit index for image name for ex 000001 instead of 1
            time_step_index_str = str(time_step_index)
            time_step_index_str =\
                (6-len(time_step_index_str))*"0" + time_step_index_str
            SaveScreenshot(save + '_' + time_step_index_str +
                           '.png', current_view)
    return None


def display_time(xdmf_reader, render_view):
    annotate_time = AnnotateTimeFilter(xdmf_reader)
    annotate_time.Format = 'Time: %f s'
    annotate_time_display = Show(annotate_time, render_view)
    annotate_time_display.WindowLocation = 'UpperCenter'
    annotate_time_display.FontFamily = 'Courier'
    annotate_time_display.FontSize = 13
    render_view.Update()
    return None


def main(args):
    '''
    :param args: command line arguments
    Render image(s) based on parameters specified
    '''

    # Disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Get data from file, set default camera properties
    input_file = load_input_file(args["input_file"])
    render_view, xdmf_reader, xdmf_reader_display = read_xdmf(
        input_file.pv_file_path)
    set_default_camera(render_view)

    # Set scalar and vector variables from input files
    # Create different source objects for scalar and vector rendering
    scalar_variable = input_file.pv_scalar_variable_properties.pv_variable_name
    vector_variable = input_file.pv_vector_variable_properties.pv_variable_name
    scalar_var_source = GetActiveSource()
    vector_var_source = GetActiveSource()

    # For vector variable properties:
    # Add vector field (glyphs on ParaView):
    if input_file.pv_vector_variable_properties.pv_add_vector_field:
        render_view, vector_var_display, vector_var_source =\
        add_vector_field(vector_variable, vector_var_source, render_view)

    # Apply filters
    if input_file.pv_filters.pv_clip.pv_apply:
        render_view, scalar_var_source, scalar_var_display=\
        apply_clip(input_file.pv_filters.pv_clip,
                   scalar_variable, render_view, scalar_var_source)
        if input_file.pv_vector_variable_properties.pv_add_vector_field:
            render_view, vector_var_source, vector_var_display=\
            apply_clip(input_file.pv_filters.pv_clip,
                       vector_variable, render_view, vector_var_source)
    elif input_file.pv_filters.pv_slice.pv_apply:
        render_view, scalar_var_source, scalar_var_display=\
        apply_slice(input_file.pv_filters.pv_slice,
                    scalar_variable, render_view, scalar_var_source)
        if input_file.pv_vector_variable_properties.pv_add_vector_field:
            render_view, vector_var_source, vector_var_display=\
            apply_slice(input_file.pv_filters.pv_slice,
                        vector_variable, render_view, vector_var_source)
    render_view.Update()

    # Update Display for scalar var
    render_view, scalar_var_display=tetrahedralize(scalar_var_source,
                                                   render_view)
    render_view,scalar_var_display=set_representation(
        input_file.pv_scalar_variable_properties.pv_representation,render_view,
        scalar_var_display)
    scalar_var_display, render_view, variable_lookup_table=\
    set_color_map(scalar_variable, render_view, scalar_var_display,
                  input_file.pv_scalar_variable_properties.pv_color_map)
    render_view=\
    set_opacity(scalar_variable,
        input_file.pv_scalar_variable_properties.pv_opacity.pv_function_type,
        input_file.pv_scalar_variable_properties.pv_opacity.pv_value,
        render_view, scalar_var_source, variable_lookup_table)

    # Update Display for vector var:
    # Set color_map for vector field
    if input_file.pv_vector_variable_properties.pv_add_vector_field:
        render_view = set_vector_color_map(vector_var_display, vector_variable,
                input_file.pv_vector_variable_properties.pv_color_map,
                render_view)

    # For Warp
    if input_file.pv_warp.pv_add_warp:
        render_view = add_scalar_warp(
            scalar_variable, xdmf_reader, input_file.pv_warp.pv_scale_type,
            input_file.pv_warp.pv_scale_factor, render_view)
        if input_file.pv_vector_variable_properties.pv_add_vector_field:
            render_view = add_vector_warp(vector_variable, scalar_variable,
                xdmf_reader, input_file.pv_warp.pv_scale_type,
                input_file.pv_warp.pv_scale_factor, render_view)

    render_view.ResetCamera()
    # Save images
    save_images(render_view, xdmf_reader, args["save"])

    return None


if __name__ == "__main__":
    try:
        main(parse_cmd_line())
    except KeyboardInterrupt:
        pass
