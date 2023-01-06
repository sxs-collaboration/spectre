# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain import (deserialize_domain, deserialize_functions_of_time,
                            inertial_coordinates, ElementId)

import numpy as np
import numpy.testing as npt
import os
import spectre.IO.H5 as spectre_h5
import unittest
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr, Frame
from spectre.Informer import unit_test_src_path


class TestInertialCoordinates(unittest.TestCase):
    def test_inertial_coordinates(self):
        volfile_name = os.path.join(unit_test_src_path(),
                                    "Visualization/Python/VolTestData0.h5")
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            serialized_domain = volfile.get_domain(obs_id)
            serialized_fot = volfile.get_functions_of_time(obs_id)
        domain = deserialize_domain[3](serialized_domain)
        functions_of_time = deserialize_functions_of_time(serialized_fot)

        element_id = ElementId[3]("[B0,(L1I0,L0I0,L0I0)]")
        element_logical_coords = tnsr.I[DataVector, 3, Frame.ElementLogical](
            3 * [DataVector([-1., 1.])])
        inertial_coords = np.asarray(
            inertial_coordinates(element_logical_coords,
                                 element_id=element_id,
                                 domain=domain,
                                 time=0.,
                                 functions_of_time=functions_of_time))

        # Domain is [0, 2 pi]^3, and split in two in dimension 0
        npt.assert_allclose(
            inertial_coords,
            np.array([[0., np.pi], [0., 2 * np.pi], [0., 2 * np.pi]]))

        # Test exceptions
        with self.assertRaisesRegex(ValueError,
                                    "The 'time' argument is required"):
            inertial_coordinates(element_logical_coords,
                                 element_id=element_id,
                                 domain=domain,
                                 time=None,
                                 functions_of_time=functions_of_time)
        with self.assertRaisesRegex(
                ValueError, "The 'functions_of_time' argument is required"):
            inertial_coordinates(element_logical_coords,
                                 element_id=element_id,
                                 domain=domain,
                                 time=0.,
                                 functions_of_time=None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
