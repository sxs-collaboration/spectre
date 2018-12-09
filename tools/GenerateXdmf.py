#!/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import h5py


def generate_xdmf(file_prefix, output_filename, start_time, stop_time, stride):
    """
    Generate one XDMF file that ParaView and VisIt can use to load the data
    out of the HDF5 files.
    """
    h5files = [(h5py.File(filename, 'r'), filename)
               for filename in glob.glob(file_prefix + "*.h5")]

    element_data = h5files[0][0].get('element_data.vol')
    temporal_ids_and_values = [(x,
                                element_data.get(x).attrs['observation_value'])
                               for x in element_data.keys()]
    temporal_ids_and_values.sort(key=lambda x: x[1])

    xdmf_output = "<?xml version=\"1.0\" ?>\n" \
        "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n" \
        "<Xdmf Version=\"2.0\">\n" \
        "<Domain>\n" \
        "<Grid Name=\"Evolution\" GridType=\"Collection\" " \
        "CollectionType=\"Temporal\">\n"

    # counter used to enforce the stride
    stride_counter = 0
    for id_and_value in temporal_ids_and_values:
        if id_and_value[1] < start_time:
            continue
        if id_and_value[1] > stop_time:
            break

        stride_counter += 1
        if (stride_counter - 1) % stride != 0:
            continue

        h5temporal = element_data.get(id_and_value[0])
        xdmf_output += "  <Grid Name=\"Grids\" GridType=\"Collection\">\n"
        xdmf_output += "    <Time Value=\"%1.14e\"/>\n" % (id_and_value[1])
        # loop over each h5 file
        for h5file in h5files:
            h5temporal = h5file[0].get('element_data.vol').get(id_and_value[0])
            for grid in h5temporal.keys():
                extents = h5temporal.get(grid).attrs['extents']
                number_of_cells = 1
                for x in extents:
                    number_of_cells *= (x - 1)
                data_item = "        <DataItem Dimensions=\"%d %d %d\" " \
                    "NumberType=\"Double\" Precision=\"8\" Format=\"HDF5\">\n" \
                    % (extents[0], extents[1], extents[2])
                element_path = "          %s:/element_data.vol/%s/%s" % (
                    h5file[1], id_and_value[0], grid)
                xdmf_output += \
                    "    <Grid Name=\"%s\" GrideType=\"Uniform\">\n" % (grid)
                # Write topology information
                xdmf_output += "      <Topology TopologyType=\"Hexahedron\" " \
                    "NumberOfElements=\"%d\">\n" % (number_of_cells)
                xdmf_output += "        <DataItem Dimensions=\"%d %d %d 8\" " \
                    "NumberType=\"Int\" Format=\"HDF5\">\n" % (
                        extents[0] - 1, extents[1] - 1, extents[2] - 1)
                xdmf_output += element_path + "/connectivity\n" \
                    "        </DataItem>\n      </Topology>\n"
                # Write geometry/coordinates
                xdmf_output += "      <Geometry Type=\"X_Y_Z\">\n"
                xdmf_output += data_item + element_path + \
                    "/InertialCoordinates_x\n        </DataItem>\n"
                xdmf_output += data_item + element_path + \
                    "/InertialCoordinates_y\n        </DataItem>\n"
                xdmf_output += data_item + element_path + \
                    "/InertialCoordinates_z\n        </DataItem>\n"
                xdmf_output += "      </Geometry>\n"
                # Write tensor components as scalars
                components = list(h5temporal.get(grid).keys())
                components.remove('InertialCoordinates_x')
                components.remove('InertialCoordinates_y')
                components.remove('InertialCoordinates_z')
                components.remove('connectivity')
                for component in components:
                    xdmf_output += "      <Attribute Name=\"%s\" " \
                        "AttributeType=\"Scalar\" Center=\"Node\">\n" % (
                            component)
                    xdmf_output += data_item + element_path + (
                        "/%s\n" % component) + "        </DataItem>\n"
                    xdmf_output += "      </Attribute>\n"
                xdmf_output += "    </Grid>\n"

        # close time grid
        xdmf_output += "  </Grid>\n"

    xdmf_output += "</Grid>\n</Domain>\n</Xdmf>"

    for h5file in h5files:
        h5file[0].close()

    with open(output_filename + ".xmf", "w") as xmf_file:
        xmf_file.write(xdmf_output)


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        description="Generate XDMF file for visualizing SpECTRE data. "
        "To load the XDMF file in ParaView you must choose the 'Xdmf Reader', "
        "NOT 'Xdmf3 Reader'",
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--file-prefix',
        required=True,
        help="The common prefix of the H5 volume files to load")
    parser.add_argument(
        '--output',
        required=True,
        help="Output file name, an xmf extension will be added")
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="View only every stride'th time step")
    parser.add_argument(
        "--start-time",
        default=0.0,
        type=float,
        help="The earliest time at which to start visualizing. The start-time "
        "value is included.")
    parser.add_argument(
        "--stop-time",
        default=1e300,
        type=float,
        help="The time at which to stop visualizing. The stop-time value is "
        "not included.")
    return parser.parse_args()


if __name__ == "__main__":
    input_args = parse_args()
    generate_xdmf(input_args.file_prefix, input_args.output,
                  input_args.start_time, input_args.stop_time,
                  input_args.stride)
