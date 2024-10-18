# Distributed under the MIT License.
# See LICENSE.txt for details.

import math
import os
import unittest

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import (
    PiecewisePolynomial2,
    PiecewisePolynomial3,
    QuaternionFunctionOfTime,
    deserialize_functions_of_time,
    serialize_functions_of_time,
)
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

    def test_serialize(self):
        translation_fot = PiecewisePolynomial2(
            time=0.0,
            initial_func_and_derivs=3 * [DataVector(size=3, fill=0.0)],
            expiration_time=math.inf,
        )
        rotation_fot = QuaternionFunctionOfTime(
            time=0.0,
            initial_quat_func=[DataVector(size=4, fill=1.0)],
            initial_angle_func=4 * [DataVector(size=3, fill=0.0)],
            expiration_time=math.inf,
        )
        expansion_fot = PiecewisePolynomial3(
            time=0.0,
            initial_func_and_derivs=4 * [DataVector(size=1, fill=0.0)],
            expiration_time=math.inf,
        )
        expansion_outer_fot = PiecewisePolynomial3(
            time=0.0,
            initial_func_and_derivs=4 * [DataVector(size=1, fill=0.0)],
            expiration_time=math.inf,
        )
        not_serialized_fots = {
            "Expansion": expansion_fot,
            "ExpansionOuterBoundary": expansion_outer_fot,
            "Rotation": rotation_fot,
            "Translation": translation_fot,
        }

        serialized_fots = serialize_functions_of_time(not_serialized_fots)
        deserialized_fots = deserialize_functions_of_time(serialized_fots)
        self.assertEqual(
            not_serialized_fots["Expansion"].func_and_2_derivs(0.0),
            deserialized_fots["Expansion"].func_and_2_derivs(0.0),
        )
        self.assertEqual(
            not_serialized_fots["ExpansionOuterBoundary"].func_and_2_derivs(
                0.0
            ),
            deserialized_fots["ExpansionOuterBoundary"].func_and_2_derivs(0.0),
        )
        self.assertEqual(
            not_serialized_fots["Rotation"].func_and_2_derivs(0.0),
            deserialized_fots["Rotation"].func_and_2_derivs(0.0),
        )
        self.assertEqual(
            not_serialized_fots["Rotation"].quat_func_and_2_derivs(0.0),
            deserialized_fots["Rotation"].quat_func_and_2_derivs(0.0),
        )
        self.assertEqual(
            not_serialized_fots["Translation"].func_and_2_derivs(0.0),
            deserialized_fots["Translation"].func_and_2_derivs(0.0),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
