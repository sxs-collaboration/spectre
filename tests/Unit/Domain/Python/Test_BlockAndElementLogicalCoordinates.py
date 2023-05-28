# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr
from spectre.Domain import (
    ElementId,
    block_logical_coordinates,
    deserialize_domain,
    deserialize_functions_of_time,
    element_logical_coordinates,
)
from spectre.Informer import unit_test_src_path


class TestBlockAndElementLogicalCoordinates(unittest.TestCase):
    def test_block_and_element_logical_coordinates(self):
        volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            serialized_domain = volfile.get_domain(obs_id)
            serialized_fot = volfile.get_functions_of_time(obs_id)

        # This domain is [0, 2 pi]^3
        domain = deserialize_domain[3](serialized_domain)
        functions_of_time = deserialize_functions_of_time(serialized_fot)
        block_logical_coords = block_logical_coordinates(
            domain,
            tnsr.I[DataVector, 3, Frame.Inertial](3 * [DataVector([np.pi])]),
            time=0.0,
            functions_of_time=functions_of_time,
        )

        element_id = ElementId[3]("[B0,(L0I0,L0I0,L0I0)]")
        logical_coords = element_logical_coordinates(
            [element_id], block_logical_coords
        )
        self.assertEqual(len(logical_coords), 1)
        self.assertEqual(
            logical_coords[element_id].element_logical_coords,
            tnsr.I[DataVector, 3, Frame.ElementLogical](
                3 * [DataVector([0.0])]
            ),
        )
        self.assertEqual(logical_coords[element_id].offsets, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
