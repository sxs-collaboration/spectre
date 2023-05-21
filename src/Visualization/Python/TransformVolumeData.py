# Distributed under the MIT License.
# See LICENSE.txt for details.

import importlib
import inspect
import logging
import re
from dataclasses import dataclass
from pydoc import locate
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import click
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import (
    Frame,
    InverseJacobian,
    Jacobian,
    Scalar,
    Tensor,
    tnsr,
)
from spectre.IO.H5.IterElements import Element, iter_elements
from spectre.NumericalAlgorithms.LinearOperators import definite_integral
from spectre.Spectral import Mesh

# functools.cached_property was added in Py 3.8. Fall back to a plain
# `property` in Py 3.7.
try:
    from functools import cached_property
except ImportError:
    cached_property = property

logger = logging.getLogger(__name__)


def snake_case_to_camel_case(s: str) -> str:
    """snake_case to CamelCase"""
    return "".join([t.title() for t in s.split("_")])


def parse_pybind11_signatures(callable) -> Iterable[inspect.Signature]:
    # Pybind11 functions don't currently expose a signature that's easily
    # readable, see issue https://github.com/pybind/pybind11/issues/945.
    # Therefore, we parse the docstring. Its first line is always the function
    # signature.
    logger.debug(callable.__doc__)
    all_matches = re.findall(
        callable.__name__ + r"\((.+)\) -> (.+)", callable.__doc__
    )
    if not all_matches:
        raise ValueError(
            f"Unable to extract signature for function '{callable.__name__}'. "
            "Please make sure it is a pybind11 binding of a C++ function. "
            "If it is, please file an issue and include the following "
            "docstring:\n\n"
            + callable.__doc__
        )
    for match_args, match_ret in all_matches:
        parameters = []
        for arg in match_args.split(","):
            arg = arg.strip()
            if " = " in arg:
                arg, default_value = arg.split(" = ")
            else:
                default = inspect.Parameter.empty
            name, annotation = arg.split(": ")
            annotation = locate(annotation)
            parameters.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=annotation,
                )
            )
        yield inspect.Signature(
            parameters=parameters, return_annotation=locate(match_ret)
        )


def get_tensor_component_names(
    tensor_name: str, tensor_type: Type[Tensor]
) -> List[str]:
    """Lists all independent components like 'Vector_x', 'Vector_y', etc."""
    # Pybind11 doesn't support class methods, so `component_suffix` is a member
    # function. Construct a proxy object to call it.
    tensor_proxy = tensor_type()
    return [
        tensor_name + tensor_proxy.component_suffix(i)
        for i in range(tensor_type.size)
    ]


class KernelArg:
    """Subclasses provide data for a kernel function argument"""

    def get(self, all_tensor_data: Dict[str, Tensor], element: Element):
        """Get the data that will be passed to the kernel function"""
        raise NotImplementedError


@dataclass(frozen=True)
class TensorArg(KernelArg):
    """Kernel argument loaded from a volume file tensor dataset"""

    tensor_type: Type[Tensor]
    dataset_name: str
    extract_single_component: bool = False

    @cached_property
    def component_names(self):
        return get_tensor_component_names(self.dataset_name, self.tensor_type)

    def get(
        self, all_tensor_data: Dict[str, Tensor], element: Optional[Element]
    ):
        tensor_data = all_tensor_data[self.dataset_name]
        if element:
            components = np.asarray(tensor_data)[:, element.data_slice]
            if self.extract_single_component:
                return DataVector(components[0])
            else:
                return self.tensor_type(components)
        else:
            if self.extract_single_component:
                return tensor_data.get()
            else:
                return tensor_data


@dataclass(frozen=True)
class ElementArg(KernelArg):
    """Kernel argument retrieved from an element in the computational domain"""

    element_attr: str

    def get(self, all_tensor_data: Dict[str, Tensor], element: Element):
        return getattr(element, self.element_attr)


def parse_kernel_arg(
    arg: inspect.Parameter, map_input_names: Dict[str, str]
) -> KernelArg:
    """Determine how data for a kernel function argument will be retrieved

    Examines the name and type annotation of the argument. The following
    arguments are supported:

    - Mesh: Annotate the argument with a 'Mesh' type.
    - Coordinates: Annotate the argument with a 'tnsr.I' type and name it
        "logical_coords" / "logical_coordinates" or "inertial_coords" /
        "inertial_coordinates" / "x".
    - Jacobians: Annotate the argument with a 'Jacobian' or 'InverseJacobian'.
    - Any tensor dataset: Annotate the argument with the tensor type. By
        default, the argument name is transformed to CamelCase to determine the
        dataset name in the volume data file. Specify a mapping in
        'map_input_names' to override the default.

    For example, the following function is a possible kernel:

        def one_index_constraint(
            psi: Scalar[DataVector],
            mesh: Mesh[3],
            inv_jacobian: InverseJacobian[DataVector, 3],
        ) -> tnsr.i[DataVector, 3]:
            # ...

    The function can also be a binding of a C++ function, such as this:

        tnsr::i<DataVector, 3> one_index_constraint(
            const Scalar<DataVector>& psi,
            const Mesh<3>& mesh,
            const InverseJacobian<
              DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
              inv_jacobian) {
            // ...
        }

    Arguments:
      arg: The function argument. Must have a type annotation.
      map_input_names: A map of argument names to dataset names (optional).
        By default the argument name is transformed to CamelCase.

    Returns: A 'KernelArg' object that knows how to retrieve the data for the
      argument.
    """
    if arg.annotation is inspect.Parameter.empty:
        raise ValueError(
            "Please annotate all arguments in the kernel function. "
            f"Argument '{arg.name}' has no annotation."
        )

    if arg.annotation in Mesh.values():
        return ElementArg("mesh")
    elif arg.name in ["logical_coords", "logical_coordinates"]:
        assert arg.annotation in [
            tnsr.I[DataVector, 1, Frame.ElementLogical],
            tnsr.I[DataVector, 2, Frame.ElementLogical],
            tnsr.I[DataVector, 3, Frame.ElementLogical],
        ], (
            f"Argument '{arg.name}' has unexpected type "
            f"'{arg.annotation}'. Expected a "
            "'tnsr.I[DataVector, Dim, Frame.ElementLogical]' "
            "for logical coordinates."
        )
        return ElementArg("logical_coordinates")
    elif arg.name in ["inertial_coords", "inertial_coordinates", "x"]:
        assert arg.annotation in [
            tnsr.I[DataVector, 1, Frame.Inertial],
            tnsr.I[DataVector, 2, Frame.Inertial],
            tnsr.I[DataVector, 3, Frame.Inertial],
        ], (
            f"Argument '{arg.name}' has unexpected type "
            f"'{arg.annotation}'. Expected a "
            "'tnsr.I[DataVector, Dim, Frame.Inertial]' "
            "for inertial coordinates."
        )
        return ElementArg("inertial_coordinates")
    elif arg.annotation in [
        Jacobian[DataVector, 1],
        Jacobian[DataVector, 2],
        Jacobian[DataVector, 3],
    ]:
        return ElementArg("jacobian")
    elif arg.annotation in [
        InverseJacobian[DataVector, 1],
        InverseJacobian[DataVector, 2],
        InverseJacobian[DataVector, 3],
    ]:
        return ElementArg("inv_jacobian")
    elif arg.annotation == DataVector:
        return TensorArg(
            tensor_type=Scalar[DataVector],
            dataset_name=map_input_names.get(
                arg.name, snake_case_to_camel_case(arg.name)
            ),
            extract_single_component=True,
        )
    else:
        try:
            arg.annotation.rank
        except AttributeError:
            raise ValueError(
                f"Unknown argument type '{arg.annotation}' for argument "
                f"'{arg.name}'. It must be either a Tensor or one of "
                "the supported structural types listed in "
                "'TransformVolumeData.parse_kernel_arg'."
            )
        return TensorArg(
            tensor_type=arg.annotation,
            dataset_name=map_input_names.get(
                arg.name, snake_case_to_camel_case(arg.name)
            ),
        )


def parse_kernel_output(
    output, output_name: str, num_points: int
) -> Dict[str, Tensor]:
    """Transform the return value of a kernel to a dict of tensor datasets

    The following return values of kernels are supported:

    - Any Tensor. By default, the name of the kernel function transformed to
      CamelCase will be used as dataset name.
    - A DataVector: will be treated as a scalar.
    - A Numpy array: will be treated as a scalar or vector.
    - An int or float: will be expanded over the grid as a constant scalar.
    - A dict of the above: can be used to return multiple datasets, and to
      assign names to them. The keys of the dictionary are used as dataset
      names.
    - A list of the above: can be used to visualize output from C++ bindings
      that return a `std::array` or `std::vector`. For example, a C++ binding
      named `truncation_error` that returns 3 numbers (one per dimension) per
      element can be visualized with this. The three numbers get expanded as
      constants over the volume of the element and returned as three datasets
      named 'TruncationError(_1,_2,_3)'.

    Arguments:
      output: The return value of a kernel function.
      output_name: The dataset name. Unused if 'output' is a dict.
      num_points: Number of grid points.

    Returns: A mapping from dataset names to tensors.
    """
    # Single tensor: return directly
    try:
        output.rank
        return {output_name: output}
    except AttributeError:
        pass
    # DataVector: parse as scalar
    if isinstance(output, DataVector):
        assert len(output) == num_points, (
            f"DataVector output named '{output_name}' has size "
            f"{len(output)}, but expected size {num_points}."
        )
        return {output_name: Scalar[DataVector]([output])}
    # Numpy array: parse as scalar or vector
    if isinstance(output, np.ndarray):
        assert output.shape[-1] == num_points, (
            f"Array output named '{output_name}' has shape {output.shape}, "
            f"but expected {num_points} points in last dimension."
        )
        if output.ndim == 1 or (output.ndim == 2 and len(output) == 1):
            return {output_name: Scalar[DataVector](output)}
        elif output.ndim == 2 and len(output) in [1, 2, 3]:
            dim = len(output)
            return {output_name: tnsr.I[DataVector, dim](output)}
        else:
            raise ValueError(
                "Kernels may only return Numpy arrays representing "
                "scalars or vectors. For higher-rank tensors, construct and "
                "return the Tensor. "
                f"Kernel output named '{output_name}' has shape "
                f"{output.shape}."
            )
    # Single number: parse as constant scalar
    if isinstance(output, (int, float)):
        return {
            output_name: Scalar[DataVector](num_points=num_points, fill=output)
        }
    # Dict or other map type: parse recursively, and use keys as dataset names
    try:
        result = {}
        for name, value in output.items():
            result.update(parse_kernel_output(value, name, num_points))
        return result
    except AttributeError:
        pass
    # List or other sequence: parse recursively, and enumerate dataset names
    try:
        result = {}
        for i, value in enumerate(output):
            result.update(
                parse_kernel_output(
                    value, output_name + "_" + str(i + 1), num_points
                )
            )
        return result
    except AttributeError:
        pass
    raise ValueError(
        f"Unsupported kernel output type '{type(output)}' "
        f"(named '{output_name}'). "
        "See 'TransformVolumeData.parse_kernel_output' "
        "for supported kernel output types."
    )


class Kernel:
    def __init__(
        self,
        callable,
        map_input_names: Dict[str, str] = {},
        output_name: Optional[str] = None,
        elementwise: Optional[bool] = None,
    ):
        """Transforms volume data with a Python function

        Arguments:
          callable: A Python function that takes and returns tensors
            (from 'spectre.DataStructures.Tensor') or a limited set of
            structural information such as the mesh, coordinates, and Jacobians.
            See the 'parse_kernel_arg' function for all supported argument
            types. The function should return a single tensor, a dictionary that
            maps dataset names to tensors, or one of the other supported types
            listed in the 'parse_kernel_output' function.
          map_input_names: A map of argument names to dataset names (optional).
            By default the argument name is transformed to CamelCase.
          output_name: Name of the output dataset (optional). By default the
            function name is transformed to CamelCase. Output names for multiple
            datasets can be specified by returning a 'Dict[str, Tensor]' from
            the 'callable'.
          elementwise: Call this kernel for each element. The default is to
            call the kernel with all data in the volume file at once, unless
            element-specific data such as a Mesh or Jacobian is requested.
        """
        self.callable = callable
        # Parse function arguments
        try:
            # Try to parse as native Python function
            signature = inspect.signature(callable)
            self.args = [
                parse_kernel_arg(arg, map_input_names)
                for arg in signature.parameters.values()
            ]
        except ValueError:
            # Try to parse as pybind11 binding
            signature = None
            # The function may have multiple overloads. We select the first one
            # that works.
            overloads = list(parse_pybind11_signatures(callable))
            for overload in overloads:
                try:
                    self.args = [
                        parse_kernel_arg(arg, map_input_names)
                        for arg in overload.parameters.values()
                    ]
                    signature = overload
                except ValueError:
                    # Try the next signature
                    continue
            if signature is None:
                raise ValueError(
                    f"The function '{callable.__name__}' has no overload "
                    "with supported arguments. See "
                    "'TransformVolumeData.parse_kernel_arg' for a list of "
                    "supported arguments. The overloads are: "
                    + rich.pretty.pretty_repr(overloads)
                )
        # If any argument is not a Tensor then we have to call the kernel
        # elementwise
        if elementwise:
            self.elementwise = True
        else:
            self.elementwise = any(
                isinstance(arg, ElementArg) for arg in self.args
            )
            assert elementwise is None or elementwise == self.elementwise, (
                f"Kernel '{callable.__name__}' must be called elementwise "
                "because an argument is not a pointwise tensor."
            )
        # Use provided output name, or transform function name to CamelCase
        self.output_name = output_name or snake_case_to_camel_case(
            callable.__name__
        )

    def __call__(
        self, all_tensor_data: Dict[str, Tensor], element: Optional[Element]
    ) -> Dict[str, Tensor]:
        output = self.callable(
            *(arg.get(all_tensor_data, element) for arg in self.args)
        )
        # Get the number of grid points from the element or from any tensor
        num_points = (
            element.mesh.number_of_grid_points()
            if element
            else len(next(iter(all_tensor_data.values()))[0])
        )
        return parse_kernel_output(output, self.output_name, num_points)


def transform_volume_data(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    kernels: Sequence[Kernel],
    integrate: bool = False,
) -> Union[None, Dict[str, Sequence[float]]]:
    """Transforms data in the 'volfiles' with a sequence of 'kernels'

    Arguments:
      volfiles: Iterable of open H5 volume files, or a single H5 volume file.
        Must be opened in writable mode, e.g. in mode "a". The transformed
        data will be written back into these files. All observations in these
        files will be transformed.
      kernels: List of transformations to apply to the volume data in the form
        of 'Kernel' objects.
      integrate: Compute the volume integral over the kernels instead of
        writing them back into the volume files. The integral is computed in
        inertial coordinates for every tensor component of all kernels and over
        all observations in the volume files. For example, if a kernel returns a
        vector named 'Shift' then this function returns integrals named
        'Shift(_x,_y,_z)'. In addition, the corresponding observation values are
        returned (named 'Time'), and the inertial volume (named 'Volume').

    Returns:
      None, or the volume integrals if 'integrate' is True.
    """
    # Collect all tensor components that we need to apply the kernels
    all_tensors: Dict[str, TensorArg] = {}
    for kernel in kernels:
        for tensor_arg in kernel.args:
            # Skip non-tensors
            if not isinstance(tensor_arg, TensorArg):
                continue
            tensor_name = tensor_arg.dataset_name
            tensor_type = tensor_arg.tensor_type
            if tensor_name in all_tensors:
                assert all_tensors[tensor_name].tensor_type == tensor_type, (
                    "Two tensor arguments with the same name "
                    f"'{tensor_name}' have different types: "
                    f"'{all_tensors[tensor_name].tensor_type}' "
                    f"and '{tensor_type}'"
                )
            else:
                all_tensors[tensor_name] = tensor_arg
    logger.debug("Input datasets: " + str(list(all_tensors.keys())))

    if integrate:
        # We collect integrals in this dict and return it
        integrals: Dict[str, Sequence[float]] = {}
    else:
        # We collect output dataset names for logging so the user can find them
        output_names = set()

    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    for volfile in volfiles:
        all_observation_ids = volfile.list_observation_ids()
        num_obs = len(all_observation_ids)
        if integrate and "Time" not in integrals:
            integrals["Time"] = [
                volfile.get_observation_value(obs_id)
                for obs_id in all_observation_ids
            ]

        for i_obs, obs_id in enumerate(all_observation_ids):
            # Load tensor data for all kernels
            all_tensor_data = {
                tensor_name: tensor_arg.tensor_type(
                    np.array(
                        [
                            volfile.get_tensor_component(
                                obs_id, component_name
                            ).data
                            for component_name in tensor_arg.component_names
                        ]
                    )
                )
                for tensor_name, tensor_arg in all_tensors.items()
            }
            total_num_points = np.sum(
                np.prod(volfile.get_extents(obs_id), axis=1)
            )

            # For integrals we call the kernels elementwise, take the integral
            # over the elements, and sum up the contributions. We then skip
            # ahead to the next observation because we're not writing anything
            # back to disk.
            if integrate:
                for element in iter_elements(volfile, obs_id):
                    # Integrate volume
                    volume = integrals.setdefault("Volume", np.zeros(num_obs))
                    volume[i_obs] += definite_integral(
                        element.det_jacobian.get(), element.mesh
                    )
                    # Integrate kernels
                    for kernel in kernels:
                        transformed_tensors = kernel(all_tensor_data, element)
                        for (
                            output_name,
                            transformed_tensor,
                        ) in transformed_tensors.items():
                            for i, component in enumerate(transformed_tensor):
                                component_integral = integrals.setdefault(
                                    output_name
                                    + transformed_tensor.component_suffix(i),
                                    np.zeros(num_obs),
                                )
                                component_integral[i_obs] += definite_integral(
                                    element.det_jacobian.get() * component,
                                    element.mesh,
                                )
                continue

            # Apply kernels
            for kernel in kernels:
                # Elementwise kernels need to slice the tensor data into
                # elements, and reassemble the result into a contiguous dataset
                if kernel.elementwise:
                    transformed_tensors_data: Dict[
                        str, Tuple[np.ndarray, Type[Tensor]]
                    ] = {}
                    for element in iter_elements(volfile, obs_id):
                        transformed_tensors = kernel(all_tensor_data, element)
                        for (
                            output_name,
                            transformed_tensor,
                        ) in transformed_tensors.items():
                            transformed_tensor_data = (
                                transformed_tensors_data.setdefault(
                                    output_name,
                                    (
                                        np.zeros(
                                            (
                                                transformed_tensor.size,
                                                total_num_points,
                                            )
                                        ),
                                        type(transformed_tensor),
                                    ),
                                )
                            )[0]
                            for i, component in enumerate(transformed_tensor):
                                transformed_tensor_data[
                                    i, element.data_slice
                                ] = component
                    transformed_tensors = {
                        output_name: tensor_type(
                            transformed_tensor_data, copy=False
                        )
                        for output_name, (
                            transformed_tensor_data,
                            tensor_type,
                        ) in transformed_tensors_data.items()
                    }
                else:
                    transformed_tensors = kernel(all_tensor_data, None)

                # Write result back into volfile
                for (
                    output_name,
                    transformed_tensor,
                ) in transformed_tensors.items():
                    output_names.add(output_name)
                    for i, component in enumerate(transformed_tensor):
                        volfile.write_tensor_component(
                            obs_id,
                            component_name=(
                                output_name
                                + transformed_tensor.component_suffix(i)
                            ),
                            contiguous_tensor_data=component,
                        )
    if integrate:
        return integrals
    else:
        logger.info(f"Output datasets: {output_names}")


def parse_input_names(ctx, param, all_values):
    if all_values is None:
        return {}
    input_names = {}
    for value in all_values:
        key, value = value.split("=")
        input_names[key] = value
    return input_names


def parse_kernels(kernels, exec_files, map_input_names):
    # Load kernels from 'exec_files'
    for exec_file in exec_files:
        exec(exec_file.read(), globals(), globals())
    # Look up all kernels
    for kernel in kernels:
        if "." in kernel:
            # A module path was specified. Import function from the module
            kernel_module_path, kernel_function = kernel.rsplit(".", maxsplit=1)
            kernel_module = importlib.import_module(kernel_module_path)
            yield Kernel(
                getattr(kernel_module, kernel_function), map_input_names
            )
        else:
            # Only a function name was specified. Look up in 'globals()'.
            yield Kernel(globals()[kernel], map_input_names)


@click.command()
@click.argument(
    "h5files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help="Name of volume data subfile within each h5 file.",
)
@click.option(
    "--kernel",
    "-k",
    "kernels",
    multiple=True,
    help=(
        "Python function(s) to apply to the volume data. "
        "Specify as 'path.to.module.function_name', where the "
        "module must be available to import. "
        "Alternatively, specify just 'function_name' if the "
        "function is defined in one of the '--exec' / '-e' "
        "files."
    ),
)
@click.option(
    "--exec",
    "-e",
    "exec_files",
    type=click.File("r"),
    multiple=True,
    help=(
        "Python file(s) to execute before loading kernels. "
        "Load kernels from these files with the '--kernel' / '-k' "
        "option."
    ),
)
@click.option(
    "--input-name",
    "-i",
    "map_input_names",
    multiple=True,
    callback=parse_input_names,
    help=(
        "Map of function argument names to dataset names "
        "in the volume data file. Specify key-value pairs "
        "like 'spatial_metric=SpatialMetric'. If unspecified, "
        "the argument name is transformed to CamelCase."
    ),
)
@click.option(
    "--integrate",
    is_flag=True,
    help=(
        "Compute the volume integral over the kernels instead of "
        "writing them back into the data files. "
        "Specify '--output' / '-o' to write the integrals to "
        "a file."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help=(
        "Output file for integrals. Either a '.txt' file "
        "or a '.h5' file. Also requires the '--output-subfile' "
        "option if a '.h5' file is used. "
        "Only used if the '--integrate' flag is set."
    ),
)
@click.option(
    "--output-subfile",
    help="Subfile name in the '--output' / '-o' file, if it is an H5 file.",
)
def transform_volume_data_command(
    h5files,
    subfile_name,
    kernels,
    exec_files,
    map_input_names,
    integrate,
    output,
    output_subfile,
    **kwargs,
):
    """Transform volume data with Python functions

    Run Python functions (kernels) over all volume data in the 'H5FILES' and
    write the output data back into the same files. You can use any Python
    function as kernel that takes tensors as input arguments and returns a
    tensor (from 'spectre.DataStructures.Tensor'). The function must be
    annotated with tensor types, like this:

        def shift_magnitude(
                shift: tnsr.I[DataVector, 3],
                spatial_metric: tnsr.ii[DataVector, 3]) -> Scalar[DataVector]:
            # ...

    Any pybind11 binding of a C++ function will also work, as long as it takes
    only supported types as arguments. Supported types are tensors, as well
    structural information such as the mesh, coordinates, and Jacobians. See the
    'parse_kernel_arg' function for all supported argument types, and
    'parse_kernel_output' for all supported return types.

    The kernels can be loaded from any available Python module, such as
    'spectre.PointwiseFunctions'. You can also execute a Python file that
    defines kernels with the '--exec' / '-e' option.

    By default, the data for the input arguments are read from datasets in the
    volume files with the same names, transformed to CamelCase. For example, the
    input dataset names for the 'shift_magnitude' function above would be
    'Shift(_x,_y,_z)' and 'SpatialMetric(_xx,_yy,_zz,_xy,_xz,_yz)'.
    That is, the code uses the name 'shift' from the function argument, changes
    it to CamelCase, then reads the 'Shift(_x,_y,_z)' datasets into a
    'tnsr.I[DataVector, 3]' before passing it to the function.
    You can override the input names with the '--input-name' / '-i' option.
    The output would be written to a dataset named 'ShiftMagnitude', which is
    the function name transformed to CamelCase.
    """
    # Script should be a noop if input files are empty
    if not h5files:
        return

    open_h5_files = [
        spectre_h5.H5File(filename, "r" if integrate else "a")
        for filename in h5files
    ]

    # Print available subfile names and exit
    if not subfile_name:
        import rich.columns

        rich.print(rich.columns.Columns(open_h5_files[0].all_vol_files()))
        return

    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]
    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name

    volfiles = [h5file.get_vol(subfile_name) for h5file in open_h5_files]

    # Load kernels
    if not kernels:
        raise click.UsageError("No '--kernel' / '-k' specified.")
    kernels = list(parse_kernels(kernels, exec_files, map_input_names))

    # Apply!
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(volfiles) == 1),
    )
    task_id = progress.add_task("Applying to files")
    volfiles_progress = progress.track(volfiles, task_id=task_id)
    with progress:
        integrals = transform_volume_data(
            volfiles_progress, kernels=kernels, integrate=integrate, **kwargs
        )
        progress.update(task_id, completed=len(volfiles))

    # Write integrals to output file or print to terminal
    if integrate:
        integral_values = np.stack(list(integrals.values())).T
        integral_names = list(integrals.keys())
        if output:
            if output.endswith(".h5"):
                if not output_subfile:
                    raise click.UsageError(
                        "The '--output-subfile' option is required "
                        "when writing to H5 files."
                    )
                if not output_subfile.startswith("/"):
                    output_subfile = "/" + output_subfile
                if output_subfile.endswith(".dat"):
                    output_subfile = output_subfile[:-4]
                with spectre_h5.H5File(output, "a") as open_output_file:
                    integrals_file = open_output_file.insert_dat(
                        output_subfile, legend=integral_names, version=1
                    )
                    integrals_file.append(integral_values)
            else:
                np.savetxt(
                    output, integral_values, header=",".join(integral_names)
                )
        else:
            import rich.table

            table = rich.table.Table(*integral_names, box=None)
            for i in range(len(integral_values)):
                table.add_row(*[f"{v:g}" for v in integral_values[i]])
            rich.print(table)


if __name__ == "__main__":
    transform_volume_data_command(help_option_names=["-h", "--help"])
