# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import os
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
from spectre.support.Logging import configure_logging
from spectre.Visualization.GenerateXdmf import (
    _count_cells_in_mixed_connectivity,
    generate_xdmf,
    generate_xdmf_command,
)


class TestGenerateXdmf(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
            spectre_informer.unit_test_src_path(), "Visualization/Python"
        )
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "Visualization/GenerateXdmf",
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def write_test_volume_file(self, filename, observation_values):
        with h5py.File(filename, "w") as open_h5file:
            volfile = open_h5file.create_group("VolumeData.vol")
            volfile.attrs["dimension"] = 3
            for observation_id, observation_value in enumerate(
                observation_values
            ):
                observation = volfile.create_group(
                    f"ObservationId{observation_id}"
                )
                observation.attrs["observation_value"] = observation_value
                # New mixed-topology format: type tag 9 (Hexahedron) followed
                # by 8 vertex indices for a single 2x2x2 hex cell.
                observation.create_dataset(
                    "connectivity",
                    data=np.array([9, 0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int32),
                )
                observation.create_dataset(
                    "total_extents", data=np.array([2, 2, 2], dtype=np.int32)
                )
                observation.create_dataset(
                    "grid_names", data=np.array([b"[B0,(L0I0,L0I0,L0I0)]"])
                )
                observation.create_dataset(
                    "bases", data=np.array([0, 0, 0], dtype=np.int32)
                )
                observation.create_dataset(
                    "quadratures", data=np.array([0, 0, 0], dtype=np.int32)
                )
                observation.create_dataset(
                    "InertialCoordinates_x",
                    data=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
                )
                observation.create_dataset(
                    "InertialCoordinates_y",
                    data=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float),
                )
                observation.create_dataset(
                    "InertialCoordinates_z",
                    data=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                )
                # Cell-centered datasets for new mixed-topology format
                observation.create_dataset(
                    "ElementId", data=np.array([42], dtype=np.uint64)
                )
                observation.create_dataset(
                    "BlockId", data=np.array([0], dtype=np.uint64)
                )
                observation.create_dataset(
                    "Psi",
                    data=np.array(
                        [
                            observation_value,
                            observation_value + 1.0,
                            observation_value + 2.0,
                            observation_value + 3.0,
                            observation_value + 4.0,
                            observation_value + 5.0,
                            observation_value + 6.0,
                            observation_value + 7.0,
                        ],
                        dtype=float,
                    ),
                )

    def test_generate_xdmf(self):
        for use_tetrahedral_connectivity in [False, True]:
            data_files = glob.glob(
                os.path.join(self.data_dir, "VolTestData*.h5")
            )
            output_filename = os.path.join(
                self.test_dir, "Test_GenerateXdmf_output"
            )
            generate_xdmf(
                h5files=data_files,
                output=output_filename,
                subfile_name="element_data",
                start_time=0.0,
                stop_time=1.0,
                stride=1,
                coordinates="InertialCoordinates",
                use_tetrahedral_connectivity=use_tetrahedral_connectivity,
            )

            # The script is quite opaque right now, so we only test that we can
            # run it and it produces output without raising an error. To test
            # more details, we should refactor the script into smaller units.
            self.assertTrue(os.path.isfile(output_filename + ".xmf"))

            # Also make sure that the output doesn't change. This has caught
            # many bugs.
            # - Compare canonicalized XML with stripped whitespace. Pretty
            #   indentation was only added in Python 3.9.
            self.assertEqual(
                ET.canonicalize(
                    from_file=output_filename + ".xmf", strip_text=True
                ),
                ET.canonicalize(
                    from_file=os.path.join(
                        self.data_dir,
                        (
                            "VolTestDataTetrahedral.xmf"
                            if use_tetrahedral_connectivity
                            else "VolTestData.xmf"
                        ),
                    ),
                    strip_text=True,
                ).replace(
                    "VolTestData0.h5",
                    os.path.relpath(data_files[0], self.test_dir),
                ),
            )
            os.remove(
                os.path.join(self.test_dir, "Test_GenerateXdmf_output.xmf")
            )

    def test_surface_generate_xdmf(self):
        data_files = [os.path.join(self.data_dir, "SurfaceTestData.h5")]
        output_filename = os.path.join(
            self.test_dir, "Test_SurfaceGenerateXdmf_output"
        )
        generate_xdmf(
            h5files=data_files,
            output=output_filename,
            subfile_name="AhA",
            start_time=0.0,
            stop_time=0.03,
            stride=1,
            coordinates="InertialCoordinates",
        )

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))

        # Also make sure that the output doesn't change. This has caught
        # many bugs.
        # - Compare canonicalized XML with stripped whitespace. Pretty
        #   indentation was only added in Python 3.9.
        self.assertEqual(
            ET.canonicalize(
                from_file=output_filename + ".xmf", strip_text=True
            ),
            ET.canonicalize(
                from_file=os.path.join(self.data_dir, "SurfaceTestData.xmf"),
                strip_text=True,
            ).replace(
                "SurfaceTestData.h5",
                os.path.relpath(data_files[0], self.test_dir),
            ),
        )

    def write_test_surface_file(self, filename):
        """Create a minimal new-format 2D surface H5 file for testing.

        One spectral element with extents [3, 4] (12 nodes). Mixed
        connectivity: 3 quads (type 5) + 2 triangles (type 4) = 5 cells.
        element_id and block_id mark this as the new mixed-topology format.
        """
        with h5py.File(filename, "w") as f:
            vol = f.create_group("SurfaceData.vol")
            vol.attrs["dimension"] = 2
            obs = vol.create_group("ObservationId0")
            obs.attrs["observation_value"] = 1.0
            # total_extents for 1 element in 2D: [3, 4] → num_points = 12
            obs.create_dataset(
                "total_extents", data=np.array([3, 4], dtype=np.int32)
            )
            obs.create_dataset(
                "grid_names", data=np.array([b"[B0,(L0I0,L0I0)]"])
            )
            obs.create_dataset("bases", data=np.array([0, 0], dtype=np.int32))
            obs.create_dataset(
                "quadratures", data=np.array([0, 0], dtype=np.int32)
            )
            # Mixed connectivity: 3 quads + 2 triangles = 5 cells
            # tag 5=Quad (4 verts), tag 4=Triangle (3 verts)
            obs.create_dataset(
                "connectivity",
                data=np.array(
                    [
                        5,
                        0,
                        1,
                        4,
                        3,
                        5,
                        1,
                        2,
                        5,
                        4,
                        5,
                        3,
                        4,
                        7,
                        6,
                        4,
                        4,
                        5,
                        8,
                        4,
                        6,
                        7,
                        9,
                    ],
                    dtype=np.int32,
                ),
            )
            coords = np.linspace(0.0, 1.0, 12)
            obs.create_dataset("InertialCoordinates_x", data=coords)
            obs.create_dataset("InertialCoordinates_y", data=coords)
            obs.create_dataset("InertialCoordinates_z", data=coords)
            obs.create_dataset(
                "ElementId",
                data=np.array([10, 11, 12, 13, 14], dtype=np.uint64),
            )
            obs.create_dataset(
                "BlockId",
                data=np.array([0, 0, 0, 0, 0], dtype=np.uint64),
            )
            obs.create_dataset("Phi", data=coords)

    def test_new_format_surface_generate_xdmf(self):
        h5_file = os.path.join(self.test_dir, "NewSurface.h5")
        self.write_test_surface_file(h5_file)
        output_filename = os.path.join(self.test_dir, "Test_NewSurface_output")
        generate_xdmf(
            h5files=[h5_file],
            output=output_filename,
            subfile_name="SurfaceData",
            start_time=0.0,
            stop_time=2.0,
            stride=1,
            coordinates="InertialCoordinates",
        )
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))
        xmf_root = ET.parse(output_filename + ".xmf").getroot()

        # Should be exactly one Uniform grid (no separate pole grid)
        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 1)

        grid = uniform_grids[0]
        topo = grid.find("Topology")
        self.assertIsNotNone(topo)
        self.assertEqual(topo.attrib.get("TopologyType"), "Mixed")
        self.assertEqual(topo.attrib.get("NumberOfElements"), "5")

        attr_names = {a.attrib["Name"] for a in grid.findall("Attribute")}
        self.assertIn("ElementId", attr_names)
        self.assertIn("BlockId", attr_names)
        self.assertIn("Phi", attr_names)

        # ElementId and BlockId should be Cell centered
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] in ("ElementId", "BlockId"):
                self.assertEqual(attr.attrib.get("Center"), "Cell")
        # Phi should be Node centered
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] == "Phi":
                self.assertEqual(attr.attrib.get("Center"), "Node")

    def write_disk_h5_file(self, filename):
        """Create a minimal new-format 2D disk H5 file for testing.

        One disk element with extents [2, 3] (6 nodes). Mixed connectivity:
        2 standard quads + 1 wrapping quad + 1 center triangle = 4 cells.
        """
        with h5py.File(filename, "w") as f:
            vol = f.create_group("DiskData.vol")
            vol.attrs["dimension"] = 2
            obs = vol.create_group("ObservationId0")
            obs.attrs["observation_value"] = 0.5
            # total_extents: n_r=2, n_phi=3
            obs.create_dataset(
                "total_extents", data=np.array([2, 3], dtype=np.int32)
            )
            obs.create_dataset("grid_names", data=np.array([b"DiskMin"]))
            obs.create_dataset(
                "bases",
                data=np.array([8, 8], dtype=np.int32),  # ZernikeB2 = 8
            )
            obs.create_dataset(
                "quadratures",
                data=np.array(
                    [3, 4], dtype=np.int32
                ),  # GaussRadauUpper, Equiangular
            )
            # Mixed connectivity: 2 quads + 1 wrapping quad + 1 triangle
            # type 5 = Quad (4 verts), type 4 = Triangle (3 verts)
            obs.create_dataset(
                "connectivity",
                data=np.array(
                    [
                        5,
                        0,
                        1,
                        3,
                        2,  # Quad phi=0→1
                        5,
                        2,
                        3,
                        5,
                        4,  # Quad phi=1→2
                        5,
                        0,
                        1,
                        5,
                        4,  # wrapping Quad
                        4,
                        0,
                        2,
                        4,
                    ],  # center Triangle
                    dtype=np.int32,
                ),
            )
            coords = np.linspace(0.0, 1.0, 6)
            obs.create_dataset("InertialCoordinates_x", data=coords)
            obs.create_dataset("InertialCoordinates_y", data=coords)
            obs.create_dataset("InertialCoordinates_z", data=coords)
            obs.create_dataset(
                "ElementId",
                data=np.array([42, 42, 42, 42], dtype=np.uint64),
            )
            obs.create_dataset(
                "BlockId",
                data=np.array([0, 0, 0, 0], dtype=np.uint64),
            )
            obs.create_dataset("Phi", data=coords)

    def test_disk_generate_xdmf(self):
        h5_file = os.path.join(self.test_dir, "DiskData.h5")
        self.write_disk_h5_file(h5_file)
        output_filename = os.path.join(self.test_dir, "Test_Disk_output")
        generate_xdmf(
            h5files=[h5_file],
            output=output_filename,
            subfile_name="DiskData",
            start_time=0.0,
            stop_time=1.0,
            stride=1,
            coordinates="InertialCoordinates",
        )
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))
        xmf_root = ET.parse(output_filename + ".xmf").getroot()

        # Should be exactly one Uniform grid
        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 1)

        grid = uniform_grids[0]
        topo = grid.find("Topology")
        self.assertIsNotNone(topo)
        self.assertEqual(topo.attrib.get("TopologyType"), "Mixed")
        self.assertEqual(topo.attrib.get("NumberOfElements"), "4")

        attr_names = {a.attrib["Name"] for a in grid.findall("Attribute")}
        self.assertIn("ElementId", attr_names)
        self.assertIn("BlockId", attr_names)
        self.assertIn("Phi", attr_names)

        # ElementId and BlockId should be Cell-centered
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] in ("ElementId", "BlockId"):
                self.assertEqual(attr.attrib.get("Center"), "Cell")
        # Phi should be Node-centered
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] == "Phi":
                self.assertEqual(attr.attrib.get("Center"), "Node")

    def test_count_cells_empty(self):
        self.assertEqual(
            _count_cells_in_mixed_connectivity(np.array([], dtype=np.int32)), 0
        )

    def test_count_cells_single_wedge(self):
        # Single Wedge: tag 8 + 6 vertex indices = 7 ints → 1 cell
        conn = np.array([8, 0, 1, 2, 3, 4, 5], dtype=np.int32)
        self.assertEqual(_count_cells_in_mixed_connectivity(conn), 1)

    def test_count_cells_mixed_hex_wedge_triangle(self):
        # 2 Hex (tag 9, 8 verts each) + 3 Wedge (tag 8, 6 verts each)
        # + 1 Triangle (tag 4, 3 verts) = 6 cells
        conn = np.array(
            [
                9,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,  # Hex 1
                9,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,  # Hex 2
                8,
                0,
                1,
                2,
                4,
                5,
                6,  # Wedge 1
                8,
                1,
                2,
                3,
                5,
                6,
                7,  # Wedge 2
                8,
                2,
                3,
                4,
                6,
                7,
                8,  # Wedge 3
                4,
                0,
                1,
                2,  # Triangle
            ],
            dtype=np.int32,
        )
        self.assertEqual(_count_cells_in_mixed_connectivity(conn), 6)

    def test_count_cells_invalid_type_tag(self):
        conn = np.array([99, 0, 1, 2], dtype=np.int32)
        with self.assertRaises(ValueError):
            _count_cells_in_mixed_connectivity(conn)

    def write_wedge_h5_file(self, filename):
        """Create an H5 file with mixed Hex+Wedge connectivity."""
        with h5py.File(filename, "w") as f:
            vol = f.create_group("WedgeData.vol")
            vol.attrs["dimension"] = 3
            obs = vol.create_group("ObservationId0")
            obs.attrs["observation_value"] = 0.0
            # 6 nodes for a Wedge + 8 for a Hex = 14 nodes
            # Mixed: 1 Hex (tag 9, 8 verts) + 1 Wedge (tag 8, 6 verts)
            obs.create_dataset(
                "connectivity",
                data=np.array(
                    [
                        9,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,  # Hex (tag 9, 8 verts)
                        8,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,  # Wedge (tag 8, 6 verts)
                    ],
                    dtype=np.int32,
                ),
            )
            obs.create_dataset(
                "total_extents", data=np.array([2, 2, 2], dtype=np.int32)
            )
            obs.create_dataset(
                "grid_names", data=np.array([b"[B0,(L0I0,L0I0,L0I0)]"])
            )
            obs.create_dataset(
                "bases", data=np.array([0, 0, 0], dtype=np.int32)
            )
            obs.create_dataset(
                "quadratures", data=np.array([0, 0, 0], dtype=np.int32)
            )
            coords = np.linspace(0.0, 1.0, 14)
            obs.create_dataset("InertialCoordinates_x", data=coords)
            obs.create_dataset("InertialCoordinates_y", data=coords)
            obs.create_dataset("InertialCoordinates_z", data=coords)
            # 2 cells total
            obs.create_dataset(
                "ElementId", data=np.array([1, 2], dtype=np.uint64)
            )
            obs.create_dataset(
                "BlockId", data=np.array([0, 0], dtype=np.uint64)
            )
            obs.create_dataset("Psi", data=coords)

    def test_wedge_generate_xdmf(self):
        h5_file = os.path.join(self.test_dir, "WedgeData.h5")
        self.write_wedge_h5_file(h5_file)
        output_filename = os.path.join(self.test_dir, "Test_Wedge_output")
        generate_xdmf(
            h5files=[h5_file],
            output=output_filename,
            subfile_name="WedgeData",
            start_time=0.0,
            stop_time=1.0,
            stride=1,
            coordinates="InertialCoordinates",
        )
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))
        xmf_root = ET.parse(output_filename + ".xmf").getroot()

        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 1)

        grid = uniform_grids[0]
        topo = grid.find("Topology")
        self.assertIsNotNone(topo)
        self.assertEqual(topo.attrib.get("TopologyType"), "Mixed")
        # 1 Hex + 1 Wedge = 2 cells
        self.assertEqual(topo.attrib.get("NumberOfElements"), "2")
        # Connectivity array length: (1+8) + (1+6) = 16 ints
        data_item = topo.find("DataItem")
        self.assertIsNotNone(data_item)
        self.assertEqual(data_item.attrib.get("Dimensions"), "16")

        attr_names = {a.attrib["Name"] for a in grid.findall("Attribute")}
        self.assertIn("ElementId", attr_names)
        self.assertIn("BlockId", attr_names)
        self.assertIn("Psi", attr_names)

        # ElementId and BlockId should be Cell-centered with 2 entries
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] in ("ElementId", "BlockId"):
                self.assertEqual(attr.attrib.get("Center"), "Cell")
                di = attr.find("DataItem")
                self.assertIsNotNone(di)
                self.assertEqual(di.attrib.get("Dimensions"), "2")
        # Psi should be Node-centered
        for attr in grid.findall("Attribute"):
            if attr.attrib["Name"] == "Psi":
                self.assertEqual(attr.attrib.get("Center"), "Node")

    def test_fast_mode_multiple_files(self):
        """Fast mode with multiple files that have different grid sizes."""
        file_a = os.path.join(self.test_dir, "VolumeA.h5")
        file_b = os.path.join(self.test_dir, "VolumeB.h5")
        # File A: one 2x2x2 element (8 points, 1 cell)
        self.write_test_volume_file(file_a, [0.0, 1.0])
        # File B: different grid — one 3x2x2 element (12 points, 2 cells)
        with h5py.File(file_b, "w") as f:
            vol = f.create_group("VolumeData.vol")
            vol.attrs["dimension"] = 3
            for obs_id, obs_val in enumerate([0.0, 1.0]):
                obs = vol.create_group(f"ObservationId{obs_id}")
                obs.attrs["observation_value"] = obs_val
                obs.create_dataset(
                    "connectivity",
                    data=np.array(
                        [
                            9,
                            0,
                            1,
                            4,
                            3,
                            6,
                            7,
                            10,
                            9,
                            9,
                            1,
                            2,
                            5,
                            4,
                            7,
                            8,
                            11,
                            10,
                        ],
                        dtype=np.int32,
                    ),
                )
                obs.create_dataset(
                    "total_extents",
                    data=np.array([3, 2, 2], dtype=np.int32),
                )
                obs.create_dataset(
                    "grid_names",
                    data=np.array([b"[B1,(L0I0,L0I0,L0I0)]"]),
                )
                obs.create_dataset(
                    "bases", data=np.array([0, 0, 0], dtype=np.int32)
                )
                obs.create_dataset(
                    "quadratures",
                    data=np.array([0, 0, 0], dtype=np.int32),
                )
                coords = np.linspace(0.0, 1.0, 12)
                obs.create_dataset("InertialCoordinates_x", data=coords)
                obs.create_dataset("InertialCoordinates_y", data=coords)
                obs.create_dataset("InertialCoordinates_z", data=coords)
                obs.create_dataset(
                    "ElementId",
                    data=np.array([10, 11], dtype=np.uint64),
                )
                obs.create_dataset(
                    "BlockId",
                    data=np.array([1, 1], dtype=np.uint64),
                )
                obs.create_dataset("Psi", data=coords)

        data_files = [file_a, file_b]
        output_normal = os.path.join(self.test_dir, "multi_normal")
        output_fast = os.path.join(self.test_dir, "multi_fast")
        generate_xdmf(
            h5files=data_files,
            output=output_normal,
            subfile_name="VolumeData",
        )
        generate_xdmf(
            h5files=data_files,
            output=output_fast,
            subfile_name="VolumeData",
            fast_mode=True,
        )
        self.assertEqual(
            ET.canonicalize(from_file=output_normal + ".xmf", strip_text=True),
            ET.canonicalize(from_file=output_fast + ".xmf", strip_text=True),
        )

    def write_new_format_with_poles_h5_file(self, filename):
        """Create a new-format 2D surface H5 file that also carries a separate
        'pole_connectivity' dataset.

        The main mixed connectivity has 2 quads (2 cells) and 'ElementId' has
        one entry per main cell. The 'pole_connectivity' has 3 triangles (3
        cells), a different number of cells than the main grid.
        """
        with h5py.File(filename, "w") as f:
            vol = f.create_group("SurfaceData.vol")
            vol.attrs["dimension"] = 2
            obs = vol.create_group("ObservationId0")
            obs.attrs["observation_value"] = 1.0
            # 1 element with extents [3, 2] -> 6 nodes
            obs.create_dataset(
                "total_extents", data=np.array([3, 2], dtype=np.int32)
            )
            obs.create_dataset(
                "grid_names", data=np.array([b"[B0,(L0I0,L0I0)]"])
            )
            obs.create_dataset("bases", data=np.array([0, 0], dtype=np.int32))
            obs.create_dataset(
                "quadratures", data=np.array([0, 0], dtype=np.int32)
            )
            # Main mixed connectivity: 2 quads (tag 5, 4 verts) = 2 cells
            obs.create_dataset(
                "connectivity",
                data=np.array([5, 0, 1, 3, 2, 5, 2, 3, 5, 4], dtype=np.int32),
            )
            # Pole mixed connectivity: 3 triangles (tag 4, 3 verts) = 3 cells
            obs.create_dataset(
                "pole_connectivity",
                data=np.array(
                    [4, 0, 1, 2, 4, 2, 3, 4, 4, 4, 5, 0], dtype=np.int32
                ),
            )
            coords = np.linspace(0.0, 1.0, 6)
            obs.create_dataset("InertialCoordinates_x", data=coords)
            obs.create_dataset("InertialCoordinates_y", data=coords)
            obs.create_dataset("InertialCoordinates_z", data=coords)
            # One ElementId per main cell (2 cells)
            obs.create_dataset(
                "ElementId", data=np.array([7, 8], dtype=np.uint64)
            )
            obs.create_dataset(
                "BlockId", data=np.array([0, 0], dtype=np.uint64)
            )
            obs.create_dataset("Phi", data=coords)

    def test_new_format_poles_use_pole_cell_count(self):
        """The pole-filling topology must report the pole connectivity's own
        cell count, not the main grid's 'ElementId' cell count."""
        h5_file = os.path.join(self.test_dir, "NewSurfaceWithPoles.h5")
        self.write_new_format_with_poles_h5_file(h5_file)
        output_filename = os.path.join(
            self.test_dir, "Test_NewSurfaceWithPoles_output"
        )
        generate_xdmf(
            h5files=[h5_file],
            output=output_filename,
            subfile_name="SurfaceData",
            coordinates="InertialCoordinates",
        )
        xmf_root = ET.parse(output_filename + ".xmf").getroot()

        # One grid for the main connectivity and one for the pole filling.
        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 2)

        # Find the pole-filling grid by its connectivity dataset name.
        pole_grid = None
        for grid in uniform_grids:
            data_item = grid.find("Topology/DataItem")
            if data_item is not None and data_item.text.endswith(
                "pole_connectivity"
            ):
                pole_grid = grid
        self.assertIsNotNone(pole_grid)

        # pole_connectivity has 3 triangles, not the 2 main-grid cells.
        pole_topo = pole_grid.find("Topology")
        self.assertEqual(pole_topo.attrib.get("NumberOfElements"), "3")

    def test_changing_field_set_with_same_dataset_count(self):
        """When the set of observed fields changes between timesteps but the
        number of datasets stays the same, each timestep must list its own
        fields rather than reusing a cached field list."""
        h5_file = os.path.join(self.test_dir, "ChangingFields.h5")
        with h5py.File(h5_file, "w") as f:
            vol = f.create_group("VolumeData.vol")
            vol.attrs["dimension"] = 3
            # obs0 observes 'Psi', obs1 observes 'Phi'. Both observations have
            # the same number of datasets.
            for obs_id, (obs_val, field) in enumerate(
                [(0.0, "Psi"), (1.0, "Phi")]
            ):
                obs = vol.create_group(f"ObservationId{obs_id}")
                obs.attrs["observation_value"] = obs_val
                obs.create_dataset(
                    "connectivity",
                    data=np.array([9, 0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int32),
                )
                obs.create_dataset(
                    "total_extents", data=np.array([2, 2, 2], dtype=np.int32)
                )
                obs.create_dataset(
                    "grid_names", data=np.array([b"[B0,(L0I0,L0I0,L0I0)]"])
                )
                obs.create_dataset(
                    "bases", data=np.array([0, 0, 0], dtype=np.int32)
                )
                obs.create_dataset(
                    "quadratures", data=np.array([0, 0, 0], dtype=np.int32)
                )
                coords = np.arange(8, dtype=float)
                obs.create_dataset("InertialCoordinates_x", data=coords)
                obs.create_dataset("InertialCoordinates_y", data=coords)
                obs.create_dataset("InertialCoordinates_z", data=coords)
                obs.create_dataset(
                    "ElementId", data=np.array([42], dtype=np.uint64)
                )
                obs.create_dataset(
                    "BlockId", data=np.array([0], dtype=np.uint64)
                )
                obs.create_dataset(field, data=coords)

        output_filename = os.path.join(
            self.test_dir, "Test_ChangingFields_output"
        )
        generate_xdmf(
            h5files=[h5_file],
            output=output_filename,
            subfile_name="VolumeData",
            coordinates="InertialCoordinates",
        )
        xmf_root = ET.parse(output_filename + ".xmf").getroot()

        # One Uniform grid per timestep, in time order.
        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 2)

        first_attrs = {
            a.attrib["Name"] for a in uniform_grids[0].findall("Attribute")
        }
        second_attrs = {
            a.attrib["Name"] for a in uniform_grids[1].findall("Attribute")
        }
        self.assertIn("Psi", first_attrs)
        # The second timestep observes 'Phi', not 'Psi'.
        self.assertIn("Phi", second_attrs)
        self.assertNotIn("Psi", second_attrs)

    def test_subfile_not_found(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_subfile_not_found"
        )
        with self.assertRaisesRegex(ValueError, "Could not open subfile"):
            generate_xdmf(
                h5files=data_files,
                output=output_filename,
                subfile_name="unknown_subfile",
                start_time=0.0,
                stop_time=1.0,
                stride=1,
                coordinates="InertialCoordinates",
            )

    def test_cli(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_output"
        )
        runner = CliRunner()
        result = runner.invoke(
            generate_xdmf_command,
            [
                *data_files,
                "-o",
                output_filename,
                "-d",
                "element_data",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        # List available subfiles
        result = runner.invoke(
            generate_xdmf_command,
            [
                *data_files,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("element_data", result.output)

    def test_stride_across_multiple_files(self):
        data_files = [
            os.path.join(self.test_dir, "VolumeSegment0.h5"),
            os.path.join(self.test_dir, "VolumeSegment1.h5"),
            os.path.join(self.test_dir, "VolumeSegment2.h5"),
        ]
        self.write_test_volume_file(data_files[0], [0.0, 1000.0, 2000.0])
        self.write_test_volume_file(
            data_files[1], [2500.0, 3000.0, 3500.0, 4000.0, 4500.0]
        )
        self.write_test_volume_file(data_files[2], [4800.0])

        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_stride_output"
        )
        generate_xdmf(
            h5files=data_files,
            output=output_filename,
            subfile_name="VolumeData",
            stride=2,
        )

        xmf_root = ET.parse(output_filename + ".xmf").getroot()
        found_times = [
            float(time_element.attrib["Value"])
            for time_element in xmf_root.findall(".//Time")
        ]
        self.assertEqual(found_times, [0.0, 2000.0, 3000.0, 4000.0, 4800.0])

        # All grids should use Mixed topology
        uniform_grids = xmf_root.findall(".//Grid[@GridType='Uniform']")
        self.assertEqual(len(uniform_grids), 5)
        for grid in uniform_grids:
            topo = grid.find("Topology")
            self.assertIsNotNone(topo)
            self.assertEqual(topo.attrib.get("TopologyType"), "Mixed")
            self.assertEqual(topo.attrib.get("NumberOfElements"), "1")
            attr_names = {a.attrib["Name"] for a in grid.findall("Attribute")}
            self.assertIn("ElementId", attr_names)
            self.assertIn("BlockId", attr_names)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
