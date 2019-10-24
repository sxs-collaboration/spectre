# !/usr/bin/env python
# Distributed under the MIT License.
# See LICENSE.txt for details.

import yaml


class InputFile():
    def __init__(self, input_dict):
        '''
        Takes the dictionary outputted by the python YAML reader as an input to
        set input file data as attributes of an InputFile object
        '''
        self.pv_file_path = input_dict["File_path"]
        self.pv_scalar_variable_properties = ScalarVariableProperties(
            input_dict["Scalar_variable_properties"])
        self.pv_vector_variable_properties = VectorVariableProperties(
            input_dict["Vector_variable_properties"])
        self.pv_filters = Filters(input_dict["Filters"])
        self.pv_warp = Warp(input_dict["Warp"])
        self.pv_save_properties = SaveProperties(
            input_dict["Save_properties"])
        return None


class ScalarVariableProperties():
    def __init__(self, pv_scalar_variable_properties):
        '''
        Initializes ScalarVariableProperties object that contains the variable
        properties for the scalar variable being visualized
        '''
        self.pv_variable_name = pv_scalar_variable_properties["Variable_name"]
        self.pv_representation = pv_scalar_variable_properties["Representation"]
        self.pv_color_map = pv_scalar_variable_properties["Color_map"]
        self.pv_opacity = Opacity(pv_scalar_variable_properties["Opacity"])


class Opacity():
    def __init__(self, pv_opacity):
        '''
        Initializes Opacity object that contains the opacity function
        information for the opacity of the scalar variable being visualized
        '''
        self.pv_function_type = pv_opacity["Function_type"]
        self.pv_value = pv_opacity["Value"]


class VectorVariableProperties():
    def __init__(self,pv_vector_variable_properties):
        '''
        Initializes VectorVariableProperties object that contains the variable
        properties for the vector variable
        '''
        self.pv_add_vector_field = pv_vector_variable_properties[\
            "Add_vector_field"]
        self.pv_variable_name = pv_vector_variable_properties["Variable_name"]
        self.pv_color_map = pv_vector_variable_properties["Color_map"]


class Filters():
    def __init__(self, pv_filters):
        '''
        Initializes Filters object that contains the different filters (as
        defined in ParaView) that can be applied to the visualization.
        '''
        self.pv_clip = Clip(pv_filters["Clip"])
        self.pv_slice = Slice(pv_filters["Slice"])


class Clip():
    def __init__(self, pv_clip):
        '''
        Initializes Clip (which is a type of filter) that contains information
        to apply the filter (such as clip type: plane, box and position,...)
        '''
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


class Slice():
    def __init__(self, pv_slice):
        '''
        Initializes Slice (which is a type of filter) that contains information
        to apply the filter (such as slice type: plane, box and position,...)
        '''
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


class Warp():
    def __init__(self, pv_warp):
        '''
        Initializes the warp information - now only implemented for apply warp
        '''
        self.pv_add_warp = pv_warp["Add_warp"]
        self.pv_scale_type = pv_warp["Scale_type"]
        self.pv_scale_factor = pv_warp["Scale_factor"]

class SaveProperties():
    def __init__(self, pv_save_properties):
        '''
        Intializes SaveProperties object that contains information on the
        image(s) being saved
        '''
        self.pv_image_resolution = pv_save_properties["Image_resolution"]
