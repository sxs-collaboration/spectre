# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas.testing as pdt

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.Visualization.ReadH5 import (
    available_subfiles,
    select_observation,
    to_dataframe,
)


class TestReadH5(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(
            spectre_informer.unit_test_src_path(), "Visualization/Python"
        )

    def test_available_subfiles(self):
        # Test with open file
        with h5py.File(self.data_dir / "DatTestData.h5", "r") as open_file:
            self.assertEqual(
                available_subfiles(open_file, extension=".dat"),
                ["Group0/MemoryData.dat", "TimeSteps2.dat"],
            )
            self.assertEqual(
                available_subfiles(open_file, extension=".vol"), []
            )
        # Test with path
        self.assertEqual(
            available_subfiles(
                self.data_dir / "VolTestData0.h5", extension=".dat"
            ),
            [],
        )
        # Test with string
        self.assertEqual(
            available_subfiles(
                os.path.join(self.data_dir, "VolTestData0.h5"), extension=".vol"
            ),
            ["element_data.vol"],
        )
        # Test with multiple files
        self.assertEqual(
            available_subfiles(
                [
                    self.data_dir / "DatTestData.h5",
                    self.data_dir / "VolTestData0.h5",
                ],
                extension=".vol",
            ),
            ["element_data.vol"],
        )

    def test_to_dataframe(self):
        # h5py subfile
        with h5py.File(
            os.path.join(self.data_dir, "DatTestData.h5"), "r"
        ) as open_file:
            df = to_dataframe(open_file["TimeSteps2.dat"])
            self.assertEqual(df.columns[0], "Time")

            df_one_row = to_dataframe(
                open_file["TimeSteps2.dat"], slice=np.s_[1:]
            )
            num_rows, num_cols = df_one_row.shape
            self.assertEqual(num_rows, 1)
            self.assertEqual(num_cols, df.shape[1])
        # SpECTRE subfile
        with spectre_h5.H5File(
            os.path.join(self.data_dir, "DatTestData.h5"), "r"
        ) as open_file:
            df2 = to_dataframe(open_file.get_dat("/TimeSteps2"))
            open_file.close_current_object()
            pdt.assert_frame_equal(df2, df)
            open_file.close_current_object()

            df2_one_row = to_dataframe(
                open_file.get_dat("/TimeSteps2"), slice=np.s_[1:]
            )
            num_rows, num_cols = df2_one_row.shape
            self.assertEqual(num_rows, 1)
            self.assertEqual(num_cols, df2.shape[1])
            pdt.assert_frame_equal(df2_one_row, df_one_row)

    def test_select_observation(self):
        with spectre_h5.H5File(
            os.path.join(self.data_dir, "VolTestData0.h5"), "r"
        ) as open_file:
            open_volfile = open_file.get_vol("/element_data")
            obs_ids = open_volfile.list_observation_ids()
            expected_obs_id = obs_ids[0]
            expected_time = open_volfile.get_observation_value(expected_obs_id)
            self.assertEqual(
                select_observation(open_volfile, step=0),
                (expected_obs_id, expected_time),
            )
            self.assertEqual(
                select_observation(open_volfile, time=0.05),
                (expected_obs_id, expected_time),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
