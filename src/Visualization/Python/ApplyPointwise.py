# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import importlib
import inspect
import logging
import numpy as np
import re
import spectre.IO.H5 as spectre_h5
from pydoc import locate
from typing import Union, Iterable, Dict, Optional, Sequence, List

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
                               tensor_type: type) -> List[str]:
    """Lists all independent components like 'Vector_x', 'Vector_y', etc."""
    # Pybind11 doesn't support class methods, so `component_suffix` is a member
    # function. Construct a proxy object to call it.
    tensor_proxy = tensor_type()
    return [
        tensor_name + tensor_proxy.component_suffix(i)
        for i in range(tensor_type.size)
    ]


class Kernel:
    def __init__(self,
                 callable,
                 map_input_names: Dict[str, str] = {},
                 output_name: Optional[str] = None):
        """Specifies input and output tensor names for a pointwise function

        Arguments:
          callable: A Python function that takes and returns tensors
            (from 'spectre.DataStructures.Tensor').
          map_input_names: A map of argument names to dataset names (optional).
            By default the argument name is transformed to CamelCase.
          output_name: Name of the output dataset (optional). By default the
            function name is transformed to CamelCase.
        """
        self.callable = callable
        # Get input tensor names by transforming the function argument names to
        # CamelCase, or look them up in 'map_input_names'
        try:
            signature = inspect.signature(callable)
        except ValueError:
            signature = parse_pybind11_signature(callable)
        callable_args = signature.parameters
        self.input_names = [
            map_input_names.get(arg_name, snake_case_to_camel_case(arg_name))
            for arg_name in callable_args
        ]
        # Look up tensor types for input arguments
        self.input_types = []
        for arg in callable_args.values():
            tensor_type = arg.annotation
            if tensor_type is inspect.Parameter.empty:
                raise ValueError(
                    "Please annotate all arguments in the kernel function "
                    f"'{callable}'. Argument '{arg.name}' has no annotation.")
            try:
                tensor_type.rank
            except AttributeError:
                raise ValueError(
                    "All annotations must be tensor types in the "
                    f"kernel function '{callable}'. Argument '{arg.name}' has "
                    f"type '{tensor_type}'.")
            self.input_types.append(tensor_type)
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
    all_tensors = {}
    for kernel in kernels:
        for tensor_name, tensor_type in zip(kernel.input_names,
                                            kernel.input_types):
            if tensor_name in all_tensors:
                assert all_tensors[tensor_name][0] == tensor_type, (
                    "Two tensor arguments with the same name "
                    f"'{tensor_name}' have different types: "
                    f"'{all_tensors[tensor_name][0]}' and '{tensor_type}'")
            else:
                all_tensors[tensor_name] = (tensor_type,
                                            get_tensor_component_names(
                                                tensor_name, tensor_type))
    logger.debug("Input datasets: " + str(list(all_tensors.keys())))
    logger.info("Output datasets: " +
                str([kernel.output_name for kernel in kernels]))

    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    for volfile in volfiles:
        for obs_id in volfile.list_observation_ids():
            # Load tensor data for all kernels
            all_tensor_data = {
                tensor_name: tensor_type(
                    np.array([
                        volfile.get_tensor_component(obs_id,
                                                     component_name).data
                        for component_name in component_names
                    ]))
                for tensor_name, (tensor_type,
                                  component_names) in all_tensors.items()
            }

            # Apply kernels
            # We currently only apply strictly pointwise kernels that need no
            # information about the element or mesh. To support derivatives we
            # can slice the tensor data into elements and call those kernels
            # that request it element-wise, passing them the mesh and Jacobian
            # as well.
            for kernel in kernels:
                kernel_args = [
                    all_tensor_data[input_name]
                    for input_name in kernel.input_names
                ]

                transformed_tensor = kernel.callable(*kernel_args)

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
    exclusively tensors as arguments. The kernels can be loaded from any
    available Python module, such as 'spectre.PointwiseFunctions'. You can also
    execute a Python file that defines kernels with the '--exec' / '-e' option.

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
        subfile_name = subfile_name.rstrip(".vol")
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
