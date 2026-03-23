# Distributed under the MIT License.
# See LICENSE.txt for details.

import re

import click

from spectre.Domain import ElementId


def _detect_dimension(element_id_str: str) -> int:
    """Detect the spatial dimension from an element ID string.

    Counts the number of 'LxIy' segments to determine the dimension.
    """
    segments = re.findall(r"L\d+I\d+", element_id_str)
    dim = len(segments)
    if dim < 1 or dim > 3:
        raise click.ClickException(
            f"Cannot determine dimension from '{element_id_str}'. "
            f"Expected 1-3 'LxIy' segments, found {dim}."
        )
    return dim


def element_id_to_short_id(element_id_str: str) -> int:
    """Convert an element ID string to a compact short ID.

    Auto-detects the dimension from the number of 'LxIy' segments in the
    string.
    """
    dim = _detect_dimension(element_id_str)
    return ElementId[dim](element_id_str).to_short_id()


@click.command(name="element-id-to-paraview")
@click.argument("element_ids", nargs=-1, required=True)
def element_id_to_paraview_command(element_ids):
    """Convert SpECTRE element ID strings to compact numeric IDs.

    These short IDs strip block_id, grid_index, and direction bits, keeping
    only the segment ID portion. They are useful for filtering or selecting
    elements in ParaView.

    Example usage:

        spectre element-id-to-paraview "[B2,(L2I3,L1I0,L1I1)]"
    """
    dims = {eid: _detect_dimension(eid) for eid in element_ids}
    unique_dims = set(dims.values())
    if len(unique_dims) > 1:
        mismatches = "\n".join(f"  {eid} -> {d}D" for eid, d in dims.items())
        raise click.ClickException(
            "All element IDs must have the same dimension, but got "
            f"dimensions {sorted(unique_dims)}:\n{mismatches}"
        )
    for eid in element_ids:
        click.echo(element_id_to_short_id(eid))
