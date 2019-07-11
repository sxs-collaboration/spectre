# !/usr/bin/env python
# Distributed under the MIT License.
# See LICENSE.txt for details.

import yaml


class InputFile():
    def __init__(self, input_stream):
        self.pv_file_path = input_stream["file_path"]
        self.pv_variable_properties = VariableProperties(
            input_stream["Variable_properties"])
        self.pv_filters = Filters(input_stream["Filter"])
        self.pv_warp = Warp(input_stream["Warp"])
        self.pv_save_properties = SaveProperties(
            input_stream["Save_properties"])
        return None


class VariableProperties(InputFile):
    def __init__(self, pv_variable_properties):
        self.pv_variable_name = pv_variable_properties["Variable_name"]
        self.pv_representation = pv_variable_properties["Representation"]
        self.pv_color_map = pv_variable_properties["Color_map"]
        self.pv_opacity = Opacity(pv_variable_properties["Opacity"])


class Opacity(VariableProperties):
    def __init__(self, pv_opacity):
        self.pv_function_type = pv_opacity["Function_type"]
        self.pv_value = pv_opacity["Value"]


class Filters(InputFile):
    def __init__(self, pv_filters):
        self.pv_clip = Clip(pv_filters["Clip"])
        self.pv_slice = Slice(pv_filters["Slice"])


class Clip(Filters):
    def __init__(self, pv_clip):
        self.pv_apply = pv_clip["Apply"]
        self.pv_type = pv_clip["Clip_type"]
        # For plane
        self.pv_origin = pv_clip["Origin"]
        self.pv_normal = pv_clip["Normal"]
        # For box
        self.pv_position = pv_clip["Position"]
        self.pv_rotation = pv_clip["Rotation"]
        self.pv_scale = pv_clip["Scale"]
        # For sphere
        self.pv_sphere_center = pv_clip["Center_s"]
        self.pv_sphere_radius = pv_clip["Radius_s"]
        # For cylinder
        self.pv_cylinder_center = pv_clip["Center_c"]
        self.pv_cylinder_radius = pv_clip["Radius_c"]
        self.pv_axis = pv_clip["Axis"]


class Slice(Filters):
    def __init__(self, pv_slice):
        self.pv_apply = pv_slice["Apply"]
        self.pv_type = pv_slice["Slice_type"]
        # For plane
        self.pv_origin = pv_slice["Origin"]
        self.pv_normal = pv_slice["Normal"]
        # For box
        self.pv_position = pv_slice["Position"]
        self.pv_rotation = pv_slice["Rotation"]
        self.pv_scale = pv_slice["Scale"]
        # For sphere
        self.pv_sphere_center = pv_slice["Center_s"]
        self.pv_sphere_radius = pv_slice["Radius_s"]
        # For cylinder
        self.pv_cylinder_center = pv_slice["Center_c"]
        self.pv_cylinder_radius = pv_slice["Radius_c"]
        self.pv_axis = pv_slice["Axis"]


class Warp(InputFile):
    def __init__(self, pv_warp):
        self.pv_add_warp = pv_warp["Add_warp"]


class SaveProperties(InputFile):
    def __init__(self, pv_save_properties):
        self.pv_image_resolution = pv_save_properties["Image_resolution"]
