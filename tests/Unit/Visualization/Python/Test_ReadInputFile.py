# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest
from pathlib import Path

import yaml

from spectre.Informer import unit_test_src_path
from spectre.Visualization.ReadInputFile import find_event, find_phase_change


class TestReadInputFile(unittest.TestCase):
    def setUp(self):
        self.input_files_dir = Path(unit_test_src_path(), "..", "InputFiles")
        self.input_file = Path(self.input_files_dir, "Burgers/Step.yaml")

    def test_find_event(self):
        with self.input_file.open() as open_input_file:
            _, input_file = yaml.safe_load_all(open_input_file)
        self.assertEqual(
            find_event("ChangeSlabSize", input_file)["DelayChange"], 5
        )
        self.assertIsNone(find_event("NonexistentEvent", input_file))

    def test_find_phase_change(self):
        with self.input_file.open() as open_input_file:
            _, input_file = yaml.safe_load_all(open_input_file)
        self.assertEqual(
            find_phase_change("VisitAndReturn(LoadBalancing)", input_file), {}
        )
        self.assertIsNone(
            find_phase_change("NonexistentPhaseChange", input_file)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
