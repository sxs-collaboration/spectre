# Distributed under the MIT License.
# See LICENSE.txt for details.

import importlib
import inspect
import logging
import re
from dataclasses import dataclass
from pydoc import locate
from typing import Dict, Iterable, List, Optional, Sequence, Type, Union

import click
import numpy as np
import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import (Frame, InverseJacobian, Jacobian,
                                           Tensor, tnsr)
from spectre.IO.H5.IterElements import Element, iter_elements
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


def parse_pybind11_signature(callable) -> inspect.Signature:
    # Pybind11 functions don't currently expose a signature that's easily
    # readable, see issue https://github.com/pybind/pybind11/issues/945.
    # Therefore, we parse the docstring. Its first line is always the function
    # signature.
    match = re.match(callable.__name__ + r"\((.+)\) -> (.+)", callable.__doc__)
    if not match:
        raise ValueError(
            f"Unable to extract signature for function '{callable.__name__}'. "
            "Please make sure it is a pybind11 binding of a C++ function. "
            "If it is, please file an issue and include the following first "
            "line of its docstring:\n\n" + callable.__doc__.partition("\n")[0])
    match_args, match_ret = match.groups()
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
            inspect.Parameter(name=name,
                              kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=default,
                              annotation=annotation))
    return inspect.Signature(parameters=parameters,
                             return_annotation=locate(match_ret))


def get_tensor_component_names(tensor_name: str,
                               tensor_type: Type[Tensor]) -> List[str]:
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

    @cached_property
    def component_names(self):
        return get_tensor_component_names(self.dataset_name, self.tensor_type)

    def get(self, all_tensor_data: Dict[str, Tensor],
            element: Optional[Element]):
        tensor_data = all_tensor_data[self.dataset_name]
        if element:
            components = np.asarray(tensor_data)[:, element.data_slice]
            return self.tensor_type(components)
        else:
            return tensor_data


@dataclass(frozen=True)
class ElementArg(KernelArg):
    """Kernel argument retrieved from an element in the computational domain"""
    element_attr: str

    def get(self, all_tensor_data: Dict[str, Tensor], element: Element):
        return getattr(element, self.element_attr)


def parse_kernel_arg(arg: inspect.Parameter,
                     map_input_names: Dict[str, str]) -> KernelArg:
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
            f"Argument '{arg.name}' has no annotation.")

    if arg.annotation in Mesh.values():
        return ElementArg("mesh")
    elif arg.name in ["logical_coords", "logical_coordinates"]:
        assert arg.annotation in [
            tnsr.I[DataVector, 1, Frame.ElementLogical],
            tnsr.I[DataVector, 2, Frame.ElementLogical],
            tnsr.I[DataVector, 3, Frame.ElementLogical],
        ], (f"Argument '{arg.name}' has unexpected type "
            f"'{arg.annotation}'. Expected a "
            "'tnsr.I[DataVector, Dim, Frame.ElementLogical]' "
            "for logical coordinates.")
        return ElementArg("logical_coordinates")
    elif arg.name in ["inertial_coords", "inertial_coordinates", "x"]:
        assert arg.annotation in [
            tnsr.I[DataVector, 1, Frame.Inertial],
            tnsr.I[DataVector, 2, Frame.Inertial],
            tnsr.I[DataVector, 3, Frame.Inertial],
        ], (f"Argument '{arg.name}' has unexpected type "
            f"'{arg.annotation}'. Expected a "
            "'tnsr.I[DataVector, Dim, Frame.Inertial]' "
            "for inertial coordinates.")
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
    else:
        try:
            arg.annotation.rank
        except AttributeError:
            raise ValueError(
                f"Unknown argument type '{arg.annotation}' for argument "
                f"'{arg.name}'. It must be either a Tensor or one of "
                "the supported structural types listed in "
                "'ApplyPointwise.parse_kernel_arg'.")
        return TensorArg(tensor_type=arg.annotation,
                         dataset_name=map_input_names.get(
                             arg.name, snake_case_to_camel_case(arg.name)))


class Kernel:
    def __init__(self,
                 callable,
                 map_input_names: Dict[str, str] = {},
                 output_name: Optional[str] = None,
                 elementwise: Optional[bool] = None):
        """Specifies input and output tensor names for a pointwise function

        Arguments:
          callable: A Python function that takes and returns tensors
            (from 'spectre.DataStructures.Tensor') or a limited set of
            structural information such as the mesh, coordinates, and Jacobians.
            See the 'parse_kernel_arg' function for all supported argument
            types.
          map_input_names: A map of argument names to dataset names (optional).
            By default the argument name is transformed to CamelCase.
          output_name: Name of the output dataset (optional). By default the
            function name is transformed to CamelCase.
          elementwise: Call this kernel for each element. The default is to
            call the kernel with all data in the volume file at once, unless
            element-specific data such as a Mesh or Jacobian is requested.
        """
        self.callable = callable
        try:
            signature = inspect.signature(callable)
        except ValueError:
            signature = parse_pybind11_signature(callable)
        # Parse function arguments
        self.args = [
            parse_kernel_arg(arg, map_input_names)
            for arg in signature.parameters.values()
        ]
        # If any argument is not a Tensor then we have to call the kernel
        # elementwise
        if elementwise:
            self.elementwise = True
        else:
            self.elementwise = any(
                isinstance(arg, ElementArg) for arg in self.args)
            assert elementwise is None or elementwise == self.elementwise, (
                f"Kernel '{callable.__name__}' must be called elementwise "
                "because an argument is not a pointwise tensor.")
        # Use provided output name, or transform function name to CamelCase
        self.output_name = (output_name
                            or snake_case_to_camel_case(callable.__name__))
        self.output_type = signature.return_annotation
        try:
            self.output_type.rank
        except AttributeError:
            raise ValueError(
                "The return annotation must be a tensor type in the "
                f"kernel function '{callable}'. Instead, it is "
                f"'{self.output_type}'.")

    def __call__(self, all_tensor_data: Dict[str, Tensor],
                 element: Optional[Element]):
        output = self.callable(*(arg.get(all_tensor_data, element)
                                 for arg in self.args))
        return output


def apply_pointwise(volfiles: Union[spectre_h5.H5Vol,
                                    Iterable[spectre_h5.H5Vol]],
                    kernels: Sequence[Kernel]):
    """Apply pointwise functions to data in the 'volfiles'

    Arguments:
      volfiles: Iterable of open H5 volume files, or a single H5 volume file.
        Must be opened in writable mode, e.g. in mode "a". The transformed
        data will be written back into these files. All observations in these
        files will be transformed.
      kernels: List of pointwise transformations to apply to the volume data
        in the form of 'Kernel' objects.
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
                    f"and '{tensor_type}'")
            else:
                all_tensors[tensor_name] = tensor_arg
    logger.debug("Input datasets: " + str(list(all_tensors.keys())))
    logger.info("Output datasets: " +
                str([kernel.output_name for kernel in kernels]))

    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    for volfile in volfiles:
        all_observation_ids = volfile.list_observation_ids()
        num_obs = len(all_observation_ids)
        for i_obs, obs_id in enumerate(all_observation_ids):
            # Load tensor data for all kernels
            all_tensor_data = {
                tensor_name: tensor_arg.tensor_type(
                    np.array([
                        volfile.get_tensor_component(obs_id,
                                                     component_name).data
                        for component_name in tensor_arg.component_names
                    ]))
                for tensor_name, tensor_arg in all_tensors.items()
            }
            total_num_points = np.sum(
                np.prod(volfile.get_extents(obs_id), axis=1))

            # Apply kernels
            for kernel in kernels:
                # Elementwise kernels need to slice the tensor data into
                # elements, and reassemble the result into a contiguous dataset
                if kernel.elementwise:
                    transformed_tensor_data = np.zeros(
                        (kernel.output_type.size, total_num_points))
                    for element in iter_elements(volfile, obs_id):
                        transformed_tensor = kernel(all_tensor_data, element)
                        for i, component in enumerate(transformed_tensor):
                            transformed_tensor_data[
                                i, element.data_slice] = component
                    transformed_tensor = kernel.output_type(
                        transformed_tensor_data, copy=False)
                else:
                    transformed_tensor = kernel(all_tensor_data, None)

                # Write result back into volfile
                for i, component in enumerate(transformed_tensor):
                    volfile.write_tensor_component(
                        obs_id,
                        component_name=(
                            kernel.output_name +
                            transformed_tensor.component_suffix(i)),
                        contiguous_tensor_data=component)


def parse_input_names(ctx, param, all_values):
    if all_values is None:
        return {}
    input_names = {}
    for value in all_values:
        key, value = value.split('=')
        input_names[key] = value
    return input_names


def parse_kernels(kernels, exec_files, map_input_names):
    # Load kernels from 'exec_files'
    for exec_file in exec_files:
        exec(exec_file.read())
    # Look up all kernels
    for kernel in kernels:
        if "." in kernel:
            # A module path was specified. Import function from the module
            kernel_module_path, kernel_function = kernel.rsplit(".",
                                                                maxsplit=1)
            kernel_module = importlib.import_module(kernel_module_path)
            yield Kernel(getattr(kernel_module, kernel_function),
                         map_input_names)
        else:
            # Only a function name was specified. Look up in 'locals()'.
            yield Kernel(locals()[kernel], map_input_names)


@click.command()
@click.argument("h5files",
                nargs=-1,
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True))
@click.option("--subfile-name",
              "-d",
              help="Name of volume data subfile within each h5 file.")
@click.option("--kernel",
              "-k",
              "kernels",
              multiple=True,
              required=True,
              help=("Python function(s) to apply to the volume data. "
                    "Specify as 'path.to.module.function_name', where the "
                    "module must be available to import. "
                    "Alternatively, specify just 'function_name' if the "
                    "function is defined in one of the '--exec' / '-e' "
                    "files."))
@click.option("--exec",
              "-e",
              "exec_files",
              type=click.File("r"),
              multiple=True,
              help=("Python file(s) to execute before loading kernels. "
                    "Load kernels from these files with the '--kernel' / '-k' "
                    "option."))
@click.option("--input-name",
              "-i",
              "map_input_names",
              multiple=True,
              callback=parse_input_names,
              help=("Map of function argument names to dataset names "
                    "in the volume data file. Specify key-value pairs "
                    "like 'spatial_metric=SpatialMetric'. If unspecified, "
                    "the argument name is transformed to CamelCase."))
def apply_pointwise_command(h5files, subfile_name, kernels, exec_files,
                            map_input_names, **kwargs):
    """Apply pointwise functions to volume data

    Run pointwise functions (kernels) over all volume data in the 'H5FILES' and
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
    'parse_kernel_arg' function for all supported argument types.

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

    open_h5_files = [spectre_h5.H5File(filename, "a") for filename in h5files]

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
    kernels = list(parse_kernels(kernels, exec_files, map_input_names))

    # Apply!
    import rich.progress
    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(volfiles) == 1))
    task_id = progress.add_task("Applying to files")
    volfiles_progress = progress.track(volfiles, task_id=task_id)
    with progress:
        apply_pointwise(volfiles_progress, kernels=kernels, **kwargs)
        progress.update(task_id, completed=len(volfiles))


if __name__ == "__main__":
    apply_pointwise_command(help_option_names=["-h", "--help"])
