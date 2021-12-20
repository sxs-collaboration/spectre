# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest
import yaml
from spectre.support.Machines import (Machine, this_machine,
                                      UnknownMachineError)
from spectre.Informer import unit_test_build_path


class TestMachines(unittest.TestCase):
    def setUp(self):
        self.machinefile_path = os.path.join(unit_test_build_path(),
                                             'TestMachine.yaml')
        yaml.safe_dump(
            dict(Machine=dict(Name="TestMachine",
                              Description="Just for testing")),
            open(self.machinefile_path, 'w'))

    def test_this_machine(self):
        with self.assertRaises(UnknownMachineError):
            this_machine('NonexistentMachinefile.yaml')
        self.assertEqual(
            this_machine(self.machinefile_path).Name, "TestMachine")


if __name__ == '__main__':
    unittest.main(verbosity=2)
