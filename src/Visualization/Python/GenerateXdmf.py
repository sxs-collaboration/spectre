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
from typing import Optional

import click
import h5py
import numpy as np
import rich

from spectre.Visualization.ReadH5 import available_subfiles

logger = logging.getLogger(__name__)


def _list_tensor_components(observation):
    components = list(observation.keys())
    components.remove("connectivity")
    if "pole_connectivity" in components:
        components.remove("pole_connectivity")
    components.remove("total_extents")
    components.remove("grid_names")
    components.remove("bases")
    components.remove("quadratures")
    if "domain" in components:
        components.remove("domain")
    if "functions_of_time" in components:
        components.remove("functions_of_time")
    return components


def _xmf_dtype(dtype: type):
    assert dtype in [
        np.dtype("float32"),
        np.dtype("float64"),
    ], f"Data type must be either a 32-bit or 64-bit float but got {dtype}."
    return "Double" if dtype == np.dtype("float64") else "Float"


def _xmf_topology(
    observation, topology_type: str, connectivity_name: str, grid_path: str
) -> ET.Element:
    num_vertices = {
        "Hexahedron": 8,
        "Quadrilateral": 4,
        "Triangle": 3,
    }[topology_type]
    num_cells = len(observation[connectivity_name]) // num_vertices
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


def _xmf_geometry(
    observation, coordinates: str, dim: int, num_points: int, grid_path: str
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
            NumberType=_xmf_dtype(observation[component_name].dtype),
            Precision="8",
            Format="HDF5",
        )
        xmf_data_item.text = os.path.join(grid_path, component_name)
    return xmf_geometry


def _xmf_scalar(
    observation, name: str, num_points: int, grid_path: str
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
        NumberType=_xmf_dtype(observation[name].dtype),
        Precision="8",
        Format="HDF5",
    )
    xmf_data_item.text = os.path.join(grid_path, name)
    return xmf_attribute


def _xmf_vector(
    observation, name: str, dim: int, num_points: int, grid_path: str
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
            NumberType=_xmf_dtype(observation[component_name].dtype),
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
    filling_poles: bool = False,
) -> ET.Element:
    # Make sure the coordinates are found in the file. We assume there should
    # always be an x-coordinate.
    assert coordinates + "_x" in observation, (
        f"No '{coordinates}_x' dataset found in '{filename}'. Existing"
        " datasets with 'Coordinates' in their name: "
        + str(
            [
                dataset_name[:-2]
                for dataset_name in observation
                if "Coordinates" in dataset_name and dataset_name.endswith("_x")
            ]
        )
    )

    # Determine dimension of embedding space by counting the number of
    # coordinate components
    dim = sum((coordinates + "_" + xyz) in observation for xyz in "xyz")

    if filling_poles:
        # Filling poles is currently supported for a 2D surface embedded in 3D
        assert "pole_connectivity" in observation and topo_dim == 2 and dim == 3

    xmf_grid = ET.Element("Grid", Name=filename, GridType="Uniform")

    # Extents in the logical directions for each element in the dataset. The
    # extents are stored in one long list with the dimension varying fast and
    # the element index varying slow.
    total_extents = observation["total_extents"]
    num_elements = len(total_extents) // topo_dim
    extents = np.reshape(total_extents, (num_elements, topo_dim), order="C")
    num_points = np.sum(np.prod(extents, axis=1))

    # Configure grid location in the H5 file
    grid_path = filename + ":/" + subfile_name + "/" + temporal_id + "/"

    # Write topology
    if topo_dim == 2 and dim == 3:
        # 2D surface embedded in 3D space
        if filling_poles:
            # Cover poles with triangles
            xmf_topology = _xmf_topology(
                observation,
                topology_type="Triangle",
                connectivity_name="pole_connectivity",
                grid_path=grid_path,
            )
        else:
            # Cover 2D surface
            xmf_topology = _xmf_topology(
                observation,
                topology_type="Quadrilateral",
                connectivity_name="connectivity",
                grid_path=grid_path,
            )
    else:
        # Cover volume
        xmf_topology = _xmf_topology(
            observation,
            topology_type={3: "Hexahedron", 2: "Quadrilateral"}[topo_dim],
            connectivity_name="connectivity",
            grid_path=grid_path,
        )
    xmf_grid.append(xmf_topology)

    # Write geometry
    xmf_grid.append(
        _xmf_geometry(
            observation,
            coordinates=coordinates,
            dim=dim,
            num_points=num_points,
            grid_path=grid_path,
        )
    )

    # Write the tensors that are to be visualized
    for component in _list_tensor_components(observation):
        if component in [coordinates + "_" + xyz for xyz in "xyz"[:dim]]:
            # Skip coordinates
            continue
        elif component.endswith("_x"):
            # Vectors
            xmf_grid.append(
                _xmf_vector(
                    observation,
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
                    observation,
                    name=component,
                    num_points=num_points,
                    grid_path=grid_path,
                )
            )
    return xmf_grid


def generate_xdmf(
    h5files,
    output: str,
    subfile_name: str,
    start_time: Optional[float] = None,
    stop_time: Optional[float] = None,
    stride: int = 1,
    coordinates: str = "InertialCoordinates",
):
    """Generate an XDMF file for ParaView and VisIt

    Read volume data from the 'H5FILES' and generate an XDMF file. The XDMF file
    points into the 'H5FILES' files so ParaView and VisIt can load the volume
    data. To process multiple files suffixed with the node number and from
    multiple segments specify a glob like 'Segment*/VolumeData*.h5'.

    To load the XDMF file in ParaView you must choose the 'Xdmf Reader', NOT
    'Xdmf3 Reader'.

    \f
    Arguments:
      h5files: List of H5 volume data files.
      output: Output filename. A '.xmf' extension is added if not present.
      subfile_name: Volume data subfile in the H5 files.
      start_time: Optional. The earliest time at which to start visualizing. The
        start-time value is included.
      stop_time: Optional. The time at which to stop visualizing. The stop-time
        value is not included.
      stride: Optional. View only every stride'th time step.
      coordinates: Optional. Name of coordinates dataset. Default:
        "InertialCoordinates".
    """
    # CLI scripts should be noops when input is empty
    if not h5files:
        return

    h5files = [(h5py.File(filename, "r"), filename) for filename in h5files]

    if not subfile_name:
        import rich.columns

        subfiles = available_subfiles(
            (h5file for h5file, _ in h5files), extension=".vol"
        )
        if len(subfiles) == 1:
            subfile_name = subfiles[0]
        else:
            rich.print(rich.columns.Columns(subfiles))
            return

    if not subfile_name.endswith(".vol"):
        subfile_name += ".vol"

    # Prepare XDMF document by building up an XML tree
    xmf_root = ET.Element("Xdmf", Version="2.0")
    xmf_domain = ET.SubElement(xmf_root, "Domain")
    xmf_timesteps = ET.SubElement(
        xmf_domain,
        "Grid",
        Name="Evolution",
        GridType="Collection",
        CollectionType="Temporal",
    )
    # Collect timesteps in a hash map before inserting into XML so we can insert
    # grids while stepping through H5 files
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

        # Sort timesteps by time
        temporal_ids_and_values = sorted(
            [
                (key, vol_subfile[key].attrs["observation_value"])
                for key in vol_subfile.keys()
            ],
            key=lambda key_and_time: key_and_time[1],
        )

        # Stride through timesteps
        for temporal_id, time in temporal_ids_and_values[::stride]:
            # Filter by start and end time
            if start_time is not None and time < start_time:
                continue
            if stop_time is not None and time > stop_time:
                break

            # A timestep is represented by a collection of grids. We store the
            # grid collection in a hash map so each H5 file can insert grids.
            if temporal_id in timesteps:
                xmf_timestep_grid = timesteps[temporal_id]
            else:
                xmf_timestep_grid = ET.SubElement(
                    xmf_timesteps, "Grid", Name="Grids", GridType="Collection"
                )
                # The time is stored as a `Time` tag in the grid collection
                ET.SubElement(xmf_timestep_grid, "Time", Value=f"{time:.14e}")
                timesteps[temporal_id] = xmf_timestep_grid

            # Construct the grid for this observation
            observation = vol_subfile[temporal_id]
            xmf_timestep_grid.append(
                _xmf_grid(
                    observation,
                    topo_dim=topo_dim,
                    filename=filename,
                    subfile_name=subfile_name,
                    temporal_id=temporal_id,
                    coordinates=coordinates,
                )
            )
            # Connect poles if the data is a 2D surface in 3D
            if "pole_connectivity" in observation:
                xmf_timestep_grid.append(
                    _xmf_grid(
                        observation,
                        topo_dim=topo_dim,
                        filename=filename,
                        subfile_name=subfile_name,
                        temporal_id=temporal_id,
                        coordinates=coordinates,
                        filling_poles=True,
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

    # Output XML
    xmf_document = """\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">
"""
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
        "not included."
    ),
)
@click.option(
    "--coordinates",
    default="InertialCoordinates",
    show_default=True,
    help="The coordinates to use for visualization",
)
def generate_xdmf_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    generate_xdmf(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_xdmf_command(help_option_names=["-h", "--help"])
