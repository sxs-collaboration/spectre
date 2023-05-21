# Distributed under the MIT License.
# See LICENSE.txt for details.

import math
import os
import unittest

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import deserialize_functions_of_time
from spectre.Informer import unit_test_src_path


class TestFunctionsOfTime(unittest.TestCase):
    def test_deserialize(self):
        volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            serialized_fot = volfile.get_functions_of_time(obs_id)

        functions_of_time = deserialize_functions_of_time(serialized_fot)
        translation = functions_of_time["Translation"]
        self.assertEqual(translation.time_bounds(), [0.0, math.inf])
        self.assertEqual(translation.func(0.0), [DataVector(size=3, fill=0.0)])
        self.assertEqual(
            translation.func_and_deriv(0.0), 2 * [DataVector(size=3, fill=0.0)]
        )
        self.assertEqual(
            translation.func_and_2_derivs(0.0),
            3 * [DataVector(size=3, fill=0.0)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
