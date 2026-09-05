# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.EccentricityControl.DirectoryStructure import (
    EccIteration,
    list_ecc_iterations,
)
from spectre.support.Logging import configure_logging


class TestDirectoryStructure(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(),
            "Pipelines/EccentricityControl/DirectoryStructure",
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ecc_iterations(self):
        # The first iteration in an empty directory starts the numbering at zero
        self.assertIsNone(EccIteration.last(self.test_dir))
        self.assertEqual(list_ecc_iterations(self.test_dir), [])
        first_iteration = EccIteration.next(self.test_dir)
        # Without any iterations yet, the first one is also the current one
        self.assertEqual(EccIteration.current(self.test_dir), first_iteration)
        self.assertEqual(
            first_iteration,
            EccIteration(path=self.test_dir / "Ecc0", id=0),
        )
        self.assertEqual(
            EccIteration.match(first_iteration.path), first_iteration
        )
        self.assertIsNone(EccIteration.match(self.test_dir / "NotAnIteration"))
        self.assertNotEqual(
            EccIteration.match(self.test_dir / "Ecc1"), first_iteration
        )

        first_iteration.path.mkdir()
        self.assertEqual(list_ecc_iterations(self.test_dir), [first_iteration])
        self.assertEqual(EccIteration.last(self.test_dir), first_iteration)

        next_iteration = EccIteration.next(self.test_dir)
        self.assertEqual(
            next_iteration, EccIteration(path=self.test_dir / "Ecc1", id=1)
        )

        # The names are not zero-padded, so they must be sorted numerically
        tenth_iteration = EccIteration(path=self.test_dir / "Ecc10", id=10)
        tenth_iteration.path.mkdir()
        next_iteration.path.mkdir()
        self.assertEqual(
            list_ecc_iterations(self.test_dir),
            [first_iteration, next_iteration, tenth_iteration],
        )
        self.assertEqual(EccIteration.last(self.test_dir), tenth_iteration)
        self.assertEqual(EccIteration.current(self.test_dir), tenth_iteration)
        self.assertEqual(EccIteration.next(self.test_dir).id, 11)

    def test_iteration_subdirs(self):
        iteration = EccIteration.next(self.test_dir)
        self.assertEqual(iteration.id_dir, self.test_dir / "Ecc0" / "ID")
        self.assertEqual(iteration.lev_dir(1), self.test_dir / "Ecc0" / "Lev1")
        self.assertEqual(
            iteration.lev_dir(-1), self.test_dir / "Ecc0" / "Lev-1"
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
