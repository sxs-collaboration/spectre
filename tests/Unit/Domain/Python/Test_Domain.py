# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain import deserialize_domain

import os
import spectre.IO.H5 as spectre_h5
import unittest
from spectre.Informer import unit_test_src_path


class TestDomain(unittest.TestCase):
    def test_deserialize(self):
        volfile_name = os.path.join(unit_test_src_path(),
                                    "Visualization/Python/VolTestData0.h5")
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            serialized_domain = volfile.get_domain(obs_id)

        domain = deserialize_domain[3](serialized_domain)
        self.assertTrue(domain.is_time_dependent())


if __name__ == '__main__':
    unittest.main(verbosity=2)
