#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Tools to generate XDMF files that ParaView and VisIt can read.

The XDMF file format is documented here:
https://xdmf.org/index.php/XDMF_Model_and_Format
"""

import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

import click
import h5py
import numpy as np
import rich

from spectre.IO.H5.ReadH5 import available_subfiles
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.support.Logging import configure_logging

logger = logging.getLogger(__name__)


_NON_TENSOR_COMPONENTS = {
    "connectivity",
    "pole_connectivity",
    "tetrahedral_connectivity",
    "total_extents",
    "grid_names",
    "bases",
    "quadratures",
    "domain",
    "functions_of_time",
    "ElementId",
    "BlockId",
}


def _xmf_dtype(dtype: type):
    assert dtype in [
        np.dtype("float32"),
        np.dtype("float64"),
    ], f"Data type must be either a 32-bit or 64-bit float but got {dtype}."
    return "Double" if dtype == np.dtype("float64") else "Float"


class _ObservationCache:
    """Hold per-grid metadata (component names, dtypes, point and cell counts,
    connectivity lengths) for one volume subfile.

    In the default mode every observation is re-read, so a changing grid or a
    changing set of observed fields is always reflected. In fast mode the
    metadata is read once from the first observation and reused for all
    subsequent observations (the caller asserts that the grid and the observed
    variables do not change over time).
    """

    def __init__(self, fast_mode: bool = False):
        self._fast_mode = fast_mode
        self._num_datasets = None
        self._components = []
        self._dtypes = {}
        self._all_dataset_names = set()
        # Grid metadata (only reused across observations in fast mode)
        self._num_points = None
        self._number_of_cells = None
        self._pole_number_of_cells = None
        self._connectivity_lengths = {}

    def update(self, observation):
        # Only fast mode reuses metadata across observations. In the default
        # mode the component names and dtypes are re-read on every observation
        # so a changing set of observed fields (even at a constant dataset
        # count) is always reflected correctly.
        if self._fast_mode and self._num_datasets is not None:
            return
        all_names = list(observation.keys())
        self._num_datasets = len(all_names)
        self._all_dataset_names = set(all_names)
        self._components = [
            name for name in all_names if name not in _NON_TENSOR_COMPONENTS
        ]
        self._dtypes = {
            name: _xmf_dtype(observation[name].dtype)
            for name in self._components
        }

    def update_grid_metadata(self, observation, topo_dim):
        """Read num_points, cell counts, and connectivity lengths.

        Re-read on every observation in the default mode; read only on the
        first call in fast mode.
        """
        if self._fast_mode and self._num_points is not None:
            return
        total_extents = observation["total_extents"]
        num_elements = len(total_extents) // topo_dim
        extents = np.reshape(total_extents, (num_elements, topo_dim), order="C")
        self._num_points = int(np.sum(np.prod(extents, axis=1)))
        if self.has_dataset("ElementId"):
            self._number_of_cells = observation["ElementId"].shape[0]
            # The new mixed-topology pole-filling grid has no per-cell
            # 'ElementId', and its cell count differs from the main grid, so it
            # must be counted from the pole connectivity itself.
            if self.has_dataset("pole_connectivity"):
                self._pole_number_of_cells = _count_cells_in_mixed_connectivity(
                    observation["pole_connectivity"][:]
                )
        for conn_name in [
            "connectivity",
            "pole_connectivity",
            "tetrahedral_connectivity",
        ]:
            if self.has_dataset(conn_name):
                self._connectivity_lengths[conn_name] = len(
                    observation[conn_name]
                )

    @property
    def num_points(self):
        return self._num_points

    @property
    def number_of_cells(self):
        return self._number_of_cells

    @property
    def pole_number_of_cells(self):
        return self._pole_number_of_cells

    def connectivity_length(self, name: str):
        return self._connectivity_lengths[name]

    @property
    def components(self):
        return self._components

    def dtype(self, name: str):
        return self._dtypes[name]

    def has_dataset(self, name: str):
        return name in self._all_dataset_names


def _xmf_topology(
    cache: _ObservationCache,
    topology_type: str,
    connectivity_name: str,
    grid_path: str,
) -> ET.Element:
    num_vertices = {
        "Hexahedron": 8,
        "Quadrilateral": 4,
        "Tetrahedron": 4,
        "Triangle": 3,
        "Wedge": 6,
    }[topology_type]
    num_cells = cache.connectivity_length(connectivity_name) // num_vertices
    xmf_topology = ET.Element(
        "Topology",
        TopologyType=topology_type,
        NumberOfElements=str(num_cells),
    )
    xmf_data_item = ET.SubElement(
        xmf_topology,
        "DataItem",
        Dimensions=f"{num_cells} {num_vertices}",
        NumberType="Int",
        Format="HDF5",
    )
    xmf_data_item.text = os.path.join(grid_path, connectivity_name)
    return xmf_topology


def _count_cells_in_mixed_connectivity(connectivity):
    """Count cells in an XDMF mixed-topology connectivity array."""
    # Map from XDMF type integer to number of vertices per cell
    verts_per_type = {2: 2, 4: 3, 5: 4, 8: 6, 9: 8}
    number_of_cells = 0
    i = 0
    while i < len(connectivity):
        type_tag = int(connectivity[i])
        n_verts = verts_per_type.get(type_tag)
        if n_verts is None:
            raise ValueError(f"Unknown XDMF topology type tag: {type_tag}")
        i += 1 + n_verts
        number_of_cells += 1
    return number_of_cells


def _xmf_mixed_topology(
    cache: _ObservationCache,
    connectivity_name: str,
    number_of_cells: int,
    grid_path: str,
) -> ET.Element:
    """Build an XDMF Mixed topology element for a new-format connectivity."""
    connectivity_length = cache.connectivity_length(connectivity_name)
    xmf_topology = ET.Element(
        "Topology",
        TopologyType="Mixed",
        NumberOfElements=str(number_of_cells),
    )
    xmf_data_item = ET.SubElement(
        xmf_topology,
        "DataItem",
        Dimensions=str(connectivity_length),
        NumberType="Int",
        Format="HDF5",
    )
    xmf_data_item.text = os.path.join(grid_path, connectivity_name)
    return xmf_topology


def _xmf_cell_attribute(
    name: str, number_of_cells: int, grid_path: str
) -> ET.Element:
    """Build an XDMF Attribute element for a cell-centered uint64 dataset."""
    xmf_attribute = ET.Element(
        "Attribute",
        Name=name,
        AttributeType="Scalar",
        Center="Cell",
    )
    xmf_data_item = ET.SubElement(
        xmf_attribute,
        "DataItem",
        Dimensions=str(number_of_cells),
        NumberType="UInt",
        Precision="8",
        Format="HDF5",
    )
    xmf_data_item.text = os.path.join(grid_path, name)
    return xmf_attribute


def _xmf_geometry(
    cache: _ObservationCache,
    coordinates: str,
    dim: int,
    num_points: int,
    grid_path: str,
) -> ET.Element:
    # The X_Y_Z and X_Y means that the x, y, and z coordinates are stored in
    # separate datasets, rather than something like interleaved.
    xmf_geometry = ET.Element(
        "Geometry", GeometryType="X_Y_Z" if dim == 3 else "X_Y"
    )
    for xyz in "xyz"[:dim]:
        component_name = coordinates + "_" + xyz
        xmf_data_item = ET.SubElement(
            xmf_geometry,
            "DataItem",
            Dimensions=str(num_points),
            NumberType=cache.dtype(component_name),
            Precision="8",
            Format="HDF5",
        )
        xmf_data_item.text = os.path.join(grid_path, component_name)
    return xmf_geometry


def _xmf_scalar(
    cache: _ObservationCache, name: str, num_points: int, grid_path: str
) -> ET.Element:
    xmf_attribute = ET.Element(
        "Attribute",
        Name=name,
        AttributeType="Scalar",
        Center="Node",
    )
    xmf_data_item = ET.SubElement(
        xmf_attribute,
        "DataItem",
        Dimensions=str(num_points),
        NumberType=cache.dtype(name),
        Precision="8",
        Format="HDF5",
    )
    xmf_data_item.text = os.path.join(grid_path, name)
    return xmf_attribute


def _xmf_vector(
    cache: _ObservationCache,
    name: str,
    dim: int,
    num_points: int,
    grid_path: str,
) -> ET.Element:
    # Write a vector using the three components that make up the vector (i.e.
    # v_x, v_y, v_z)
    xmf_attribute = ET.Element(
        "Attribute",
        Name=name,
        AttributeType="Vector",
        Center="Node",
    )
    xmf_function = ET.SubElement(
        xmf_attribute,
        "DataItem",
        Dimensions=f"{num_points} 3",
        ItemType="Function",
        # In 2d we still need a 3d dataset to have a vector because ParaView
        # only supports 3d vectors. We deal with this by making the z-component
        # all zeros.
        Function=("JOIN($0,$1,$2)" if dim == 3 else "JOIN($0,$1, 0 * $1)"),
    )
    for xyz in "xyz"[:dim]:
        component_name = name + "_" + xyz
        xmf_data_item = ET.SubElement(
            xmf_function,
            "DataItem",
            Dimensions=str(num_points),
            NumberType=cache.dtype(component_name),
            Precision="8",
            Format="HDF5",
        )
        xmf_data_item.text = os.path.join(grid_path, component_name)
    return xmf_attribute


def _xmf_grid(
    observation,
    topo_dim: int,
    filename: str,
    subfile_name: str,
    temporal_id: str,
    coordinates: str,
    cache: _ObservationCache,
    filling_poles: bool = False,
    use_tetrahedral_connectivity: bool = False,
) -> ET.Element:
    # Make sure the coordinates are found in the file. We assume there should
    # always be an x-coordinate.
    assert cache.has_dataset(coordinates + "_x"), (
        f"No '{coordinates}_x' dataset found in '{filename}'. Existing"
        " datasets with 'Coordinates' in their name: "
        + str(
            [
                dataset_name[:-2]
                for dataset_name in cache.components
                if "Coordinates" in dataset_name and dataset_name.endswith("_x")
            ]
        )
    )

    # Determine dimension of embedding space by counting the number of
    # coordinate components
    dim = sum(cache.has_dataset(coordinates + "_" + xyz) for xyz in "xyz")

    if filling_poles:
        assert (
            cache.has_dataset("pole_connectivity")
            and topo_dim == 2
            and dim == 3
        )

    xmf_grid = ET.Element("Grid", Name=filename, GridType="Uniform")

    cache.update_grid_metadata(observation, topo_dim)
    num_points = cache.num_points

    # Configure grid location in the H5 file
    grid_path = filename + ":/" + subfile_name + "/" + temporal_id + "/"

    # Detect new mixed-topology format by presence of 'ElementId' dataset
    is_new_format = cache.has_dataset("ElementId")
    number_of_cells = cache.number_of_cells if is_new_format else None

    # Write topology
    if topo_dim == 2 and dim == 3:
        # 2D surface embedded in 3D space
        if filling_poles:
            if is_new_format:
                xmf_topology = _xmf_mixed_topology(
                    cache,
                    connectivity_name="pole_connectivity",
                    number_of_cells=cache.pole_number_of_cells,
                    grid_path=grid_path,
                )
            else:
                xmf_topology = _xmf_topology(
                    cache,
                    topology_type="Triangle",
                    connectivity_name="pole_connectivity",
                    grid_path=grid_path,
                )
        elif is_new_format:
            xmf_topology = _xmf_mixed_topology(
                cache,
                connectivity_name="connectivity",
                number_of_cells=number_of_cells,
                grid_path=grid_path,
            )
        else:
            xmf_topology = _xmf_topology(
                cache,
                topology_type="Quadrilateral",
                connectivity_name="connectivity",
                grid_path=grid_path,
            )
    else:
        # Cover volume
        if use_tetrahedral_connectivity:
            topology_type = {3: "Tetrahedron", 2: "Triangle"}[topo_dim]
            xmf_topology = _xmf_topology(
                cache,
                topology_type=topology_type,
                connectivity_name="tetrahedral_connectivity",
                grid_path=grid_path,
            )
        elif is_new_format:
            xmf_topology = _xmf_mixed_topology(
                cache,
                connectivity_name="connectivity",
                number_of_cells=number_of_cells,
                grid_path=grid_path,
            )
        else:
            topology_type = {3: "Hexahedron", 2: "Quadrilateral"}[topo_dim]
            xmf_topology = _xmf_topology(
                cache,
                topology_type=topology_type,
                connectivity_name="connectivity",
                grid_path=grid_path,
            )
    xmf_grid.append(xmf_topology)

    # Write geometry
    xmf_grid.append(
        _xmf_geometry(
            cache,
            coordinates=coordinates,
            dim=dim,
            num_points=num_points,
            grid_path=grid_path,
        )
    )

    # Write the tensors that are to be visualized
    for component in cache.components:
        if component in [coordinates + "_" + xyz for xyz in "xyz"[:dim]]:
            # Skip coordinates
            continue
        elif component.endswith("_x"):
            # Vectors
            xmf_grid.append(
                _xmf_vector(
                    cache,
                    name=component[:-2],
                    dim=dim,
                    num_points=num_points,
                    grid_path=grid_path,
                )
            )
        elif component.endswith("_y") or component.endswith("_z"):
            # Skip other vector components since they're processed above
            continue
        else:
            # Treat everything else as scalars
            xmf_grid.append(
                _xmf_scalar(
                    cache,
                    name=component,
                    num_points=num_points,
                    grid_path=grid_path,
                )
            )

    # For new-format volume data, add cell-centered element_id and block_id
    # attributes (not for the pole-filling grid, which uses pole_connectivity).
    if is_new_format and not filling_poles and not use_tetrahedral_connectivity:
        for attr_name in ["ElementId", "BlockId"]:
            if cache.has_dataset(attr_name):
                xmf_grid.append(
                    _xmf_cell_attribute(attr_name, number_of_cells, grid_path)
                )

    return xmf_grid


def get_files_with_subfile(
    h5file_names: Sequence[str], subfile_name: str
) -> List[Tuple[h5py.File, str]]:
    """Get the h5files and their name that contain a subfile with the name
    subfile_name

        \f
    Arguments:
      h5file_names: List of H5 file names of files to open and check if they
        have the subfile.
      subfile_name: The name of the subfile to check for.
    """
    result = list()
    for filename in h5file_names:
        h5file = h5py.File(filename, "r")
        if subfile_name in h5file:
            result.append((h5file, filename))
    return result


def generate_xdmf(
    h5files,
    output: str,
    subfile_name: str,
    relative_paths: bool = True,
    start_time: Optional[float] = None,
    stop_time: Optional[float] = None,
    stride: int = 1,
    coordinates: str = "InertialCoordinates",
    use_tetrahedral_connectivity: bool = False,
    fast_mode: bool = False,
):
    """Generate an XDMF file for ParaView and VisIt

    Read volume data from the 'H5FILES' and generate an XDMF file. The XDMF file
    points into the 'H5FILES' files so ParaView and VisIt can load the volume
    data. To process multiple files suffixed with the node number and from
    multiple segments specify a glob like 'Segment*/VolumeData*.h5'.

    To load the XDMF file in ParaView you must choose the 'Xdmf3 Reader', NOT
    'Xdmf Reader'.

    \f
    Arguments:
      h5files: List of H5 volume data files.
      output: Output filename. A '.xmf' extension is added if not present.
      subfile_name: Volume data subfile in the H5 files.
      relative_paths: If True, use relative paths in the XDMF file (default). If
        False, use absolute paths.
      start_time: Optional. The earliest time at which to start visualizing. The
        start-time value is included.
      stop_time: Optional. The time at which to stop visualizing. The stop-time
        value is not included.
      stride: Optional. View only every stride'th time step.
      coordinates: Optional. Name of coordinates dataset. Default:
        "InertialCoordinates".
      use_tetrahedral_connectivity: Optional. Use "tetrahedral_connectivity".
        Default: False
      fast_mode: Optional. Assume the grid and observed variables do not change
        over time. Default: False
    """
    h5file_names = h5files

    if not subfile_name:
        subfiles = available_subfiles(
            h5file_names,
            extension=".vol",
        )
        if len(subfiles) == 1:
            subfile_name = subfiles[0]
            logger.info(
                f"Selected subfile {subfile_name} (the only available one)."
            )
        else:
            raise RequiredChoiceError(
                (
                    "Specify '--subfile-name' / '-d' to select a"
                    " subfile containing volume data."
                ),
                choices=subfiles,
            )

    if not subfile_name.endswith(".vol"):
        subfile_name += ".vol"

    h5files = get_files_with_subfile(h5file_names, subfile_name)

    if len(h5files) == 0:
        raise ValueError(
            f"Could not open subfile name '{subfile_name}' in any h5 "
            f"files, {h5file_names}. Available subfiles: "
            + str(
                available_subfiles(
                    h5file_names,
                    extension=".vol",
                )
            )
        )

    # Prepare XDMF document by building up an XML tree
    xmf_root = ET.Element("Xdmf", Version="3.0")
    xmf_domain = ET.SubElement(xmf_root, "Domain")
    xmf_timesteps = ET.SubElement(
        xmf_domain,
        "Grid",
        Name="Evolution",
        GridType="Collection",
        CollectionType="Temporal",
    )
    # Collect timestep records from all input files so stride can be applied
    # globally rather than independently per file.
    timesteps = dict()

    for h5file, filename in h5files:
        # Open subfile
        try:
            vol_subfile = h5file[subfile_name]
        except KeyError as err:
            raise ValueError(
                f"Could not open subfile name '{subfile_name}' in '{filename}'."
                " Available subfiles: "
                + str(available_subfiles(h5file, extension=".vol"))
            ) from err
        topo_dim = int(vol_subfile.attrs["dimension"])
        if topo_dim == 1:
            raise ValueError(
                "The spatial dimension of the data in subfile"
                f" {subfile_name} of HDF5 file {filename} is 1d "
                "but generate-xdmf only works on 2d and 3d data."
            )

        # Use paths relative to the output file or absolute paths
        filename_in_output = (
            os.path.relpath(
                filename, os.path.dirname(output) if output else None
            )
            if relative_paths
            else os.path.abspath(filename)
        )

        # Sort timesteps by time
        temporal_ids_and_values = sorted(
            [
                (key, vol_subfile[key].attrs["observation_value"])
                for key in vol_subfile.keys()
                if key.startswith("ObservationId")
            ],
            key=lambda key_and_time: key_and_time[1],
        )

        for temporal_id, time in temporal_ids_and_values:
            timestep_key = (time, temporal_id)
            if timestep_key not in timesteps:
                timesteps[timestep_key] = []
            timesteps[timestep_key].append(
                (vol_subfile, topo_dim, filename_in_output)
            )

    # Sort timesteps globally by time and apply stride to the global sequence.
    caches = {}
    sorted_timestep_items = sorted(
        timesteps.items(), key=lambda item: (item[0][0], item[0][1])
    )
    for (time, temporal_id), timestep_records in sorted_timestep_items[
        ::stride
    ]:
        # Filter by start and end time
        if start_time is not None and time < start_time:
            continue
        if stop_time is not None and time > stop_time:
            break

        xmf_timestep_grid = ET.SubElement(
            xmf_timesteps, "Grid", Name="Grids", GridType="Collection"
        )
        # The time is stored as a `Time` tag in the grid collection
        ET.SubElement(xmf_timestep_grid, "Time", Value=f"{time:.14e}")

        for vol_subfile, topo_dim, filename_in_output in timestep_records:
            # Construct the grid for this observation.
            # Each subfile gets its own cache because different files
            # have different grid partitions (num_points, cells, etc.).
            subfile_id = id(vol_subfile)
            if subfile_id not in caches:
                caches[subfile_id] = _ObservationCache(fast_mode=fast_mode)
            cache = caches[subfile_id]
            if fast_mode and cache._num_datasets is not None:
                observation = None
            else:
                observation = vol_subfile[temporal_id]
                cache.update(observation)
            xmf_timestep_grid.append(
                _xmf_grid(
                    observation,
                    topo_dim=topo_dim,
                    filename=filename_in_output,
                    subfile_name=subfile_name,
                    temporal_id=temporal_id,
                    coordinates=coordinates,
                    cache=cache,
                    use_tetrahedral_connectivity=use_tetrahedral_connectivity,
                )
            )
            # Backwards compatibility: old files have a separate
            # 'pole_connectivity' dataset with Triangle cells to fill the poles.
            if cache.has_dataset("pole_connectivity"):
                xmf_timestep_grid.append(
                    _xmf_grid(
                        observation,
                        topo_dim=topo_dim,
                        filename=filename_in_output,
                        subfile_name=subfile_name,
                        temporal_id=temporal_id,
                        coordinates=coordinates,
                        cache=cache,
                        filling_poles=True,
                        use_tetrahedral_connectivity=(
                            use_tetrahedral_connectivity
                        ),
                    )
                )

    for h5file in h5files:
        h5file[0].close()

    # Pretty-print XML
    try:
        # Added in Py 3.9
        ET.indent(xmf_root)
    except AttributeError:
        pass

    # Output XML (XDMF 3.0 does not use the DTD declaration)
    xmf_document = '<?xml version="1.0" ?>\n'
    xmf_document += ET.tostring(xmf_root, encoding="unicode")
    xmf_document += "\n"
    if output:
        if not output.endswith(".xmf"):
            output += ".xmf"
        with open(output, "w") as open_output_file:
            open_output_file.write(xmf_document)
    else:
        sys.stdout.write(xmf_document)


@click.command(name="generate-xdmf", help=generate_xdmf.__doc__)
@click.argument(
    "h5files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help=(
        "Output file name. A '.xmf' extension will be added if not present. "
        "If unspecified, the output will be written to stdout."
    ),
)
@click.option(
    "--subfile-name",
    "-d",
    help=(
        "Name of the volume data subfile in the H5 files. A '.vol' extension is"
        " added if needed. If unspecified, and the first H5 file contains only"
        " a single '.vol' subfile, choose that. Otherwise, list all '.vol'"
        " subfiles and exit."
    ),
)
@click.option(
    "--relative-paths/--absolute-paths",
    default=True,
    show_default=True,
    help="Use relative paths or absolute paths in the XDMF file.",
)
@click.option(
    "--stride", default=1, type=int, help="View only every stride'th time step"
)
@click.option(
    "--start-time",
    type=float,
    help=(
        "The earliest time at which to start visualizing. The start-time "
        "value is included."
    ),
)
@click.option(
    "--stop-time",
    type=float,
    help=(
        "The time at which to stop visualizing. The stop-time value is "
        "included."
    ),
)
@click.option(
    "--coordinates",
    default="InertialCoordinates",
    show_default=True,
    help="The coordinates to use for visualization",
)
@click.option(
    "--use-tetrahedral-connectivity",
    is_flag=True,
    default=False,
    help=(
        "Use a tetrahedral connectivity called tetrahedral_connectivity in "
        "the HDF5 file. See the generate-tetrahedral-connectivity CLI for "
        "information on how to generate tetrahedral connectivity and what it "
        "can be useful for."
    ),
)
@click.option(
    "--fast-mode",
    is_flag=True,
    default=False,
    help=(
        "Assume the grid structure and observed variables do not change "
        "over time. Reads HDF5 metadata only once and reuses it for all "
        "timesteps. This is much faster on slow filesystems, but will "
        "produce incorrect output if the simulation uses adaptive mesh "
        "refinement (AMR) or changes which variables are observed."
    ),
)
def generate_xdmf_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    generate_xdmf(**kwargs)


if __name__ == "__main__":
    configure_logging(log_level=logging.INFO)
    generate_xdmf_command(help_option_names=["-h", "--help"])
