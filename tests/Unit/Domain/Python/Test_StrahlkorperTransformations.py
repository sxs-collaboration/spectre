# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import spectre.IO.H5 as spectre_h5
from spectre.Domain import (
    deserialize_domain,
    deserialize_functions_of_time,
    strahlkorper_in_inertial_frame,
)
from spectre.Informer import unit_test_src_path
from spectre.SphericalHarmonics import Frame, Strahlkorper


class TestStrahlkorperTransformations(unittest.TestCase):
    def test_strahlkorper_in_different_frame(self):
        volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            domain = deserialize_domain[3](volfile.get_domain())
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id)
            )

        strahlkorper_grid = Strahlkorper[Frame.Grid](
            l_max=2, radius=0.5, center=[0.5, 0.5, 0.5]
        )
        strahlkorper_inertial = strahlkorper_in_inertial_frame(
            strahlkorper_grid,
            domain=domain,
            functions_of_time=functions_of_time,
            time=0.0,
        )
        self.assertAlmostEqual(strahlkorper_inertial.average_radius, 0.5)
        strahlkorper_inertial = strahlkorper_in_inertial_frame_aligned(
            strahlkorper_grid,
            domain=domain,
            functions_of_time=functions_of_time,
            time=0.0,
        )
        self.assertAlmostEqual(strahlkorper_inertial.average_radius, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
