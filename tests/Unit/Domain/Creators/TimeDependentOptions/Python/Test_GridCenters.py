# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

from spectre.Domain.Creators.TimeDependentOptions import GridCentersOptions
from spectre.Informer import unit_test_src_path


class TestGridCenters(unittest.TestCase):
    def test_construction(self):
        grid_centers = GridCentersOptions(
            os.path.join(
                str(unit_test_src_path()),
                "../InputFiles/GrMhd/GhValenciaDivClean/",
                "EvolutionParameters.perl",
            ),
            2.0,
        )
        self.assertNotEqual(grid_centers, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
