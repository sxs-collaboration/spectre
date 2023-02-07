# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain import deserialize_domain, deserialize_functions_of_time

import numpy as np
import numpy.testing as npt
import os
import spectre.IO.H5 as spectre_h5
import unittest
from spectre.Informer import unit_test_src_path
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr


class TestDomain(unittest.TestCase):
    def test_deserialize(self):
        volfile_name = os.path.join(unit_test_src_path(),
                                    "Visualization/Python/VolTestData0.h5")
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            time = volfile.get_observation_value(obs_id)
            serialized_domain = volfile.get_domain(obs_id)
            serialized_fot = volfile.get_functions_of_time(obs_id)

        domain = deserialize_domain[3](serialized_domain)
        self.assertEqual(domain.dim, 3)
        self.assertTrue(domain.is_time_dependent())
        for block_id, block in enumerate(domain.blocks):
            self.assertEqual(block.id, block_id)
        domain.blocks[0].name == ""
        self.assertEqual(len(domain.block_groups), 0)

        # Check coordinate maps. The domain is [0, 2 pi]^3, so the block-logical
        # coord (0, 0, 0) should map to (pi,) * 3
        block = domain.blocks[0]
        functions_of_time = deserialize_functions_of_time(serialized_fot)
        logical_coord = tnsr.I[DataVector, 3, Frame.BlockLogical](num_points=1,
                                                                  fill=0.)
        self.assertTrue(block.is_time_dependent())
        self.assertFalse(block.has_distorted_frame())
        grid_coord = block.moving_mesh_logical_to_grid_map(logical_coord)
        inertial_coord = block.moving_mesh_grid_to_inertial_map(
            grid_coord, time, functions_of_time)
        npt.assert_allclose(inertial_coord, [[np.pi]] * 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
