# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest
from pathlib import Path

import yaml
from spectre.Informer import unit_test_src_path
from spectre.Visualization.ReadInputFile import find_event, get_executable


class TestReadInputFile(unittest.TestCase):
    def setUp(self):
        self.input_files_dir = Path(unit_test_src_path(), "..", "InputFiles")
        self.input_file = Path(self.input_files_dir, "Burgers/Step.yaml")

    def test_get_executable(self):
        self.assertEqual(get_executable(self.input_file.read_text()),
                         "EvolveBurgers")

    def test_find_event(self):
        with self.input_file.open() as open_input_file:
            input_file = yaml.safe_load(open_input_file)
        self.assertEqual(
            find_event("ChangeSlabSize", input_file)["DelayChange"], 5)
        self.assertIsNone(find_event("NonexistentEvent", input_file))


if __name__ == '__main__':
    unittest.main(verbosity=2)
