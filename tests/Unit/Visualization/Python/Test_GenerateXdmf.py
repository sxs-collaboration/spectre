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
                observation.create_dataset(
                    "connectivity", data=np.arange(8, dtype=np.int32)
                )
                observation.create_dataset(
                    "total_extents", data=np.array([2, 2, 2], dtype=np.int32)
                )
                observation.create_dataset(
                    "grid_names", data=np.array([b"Element0"])
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


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
