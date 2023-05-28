# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import yaml

from spectre.Informer import unit_test_build_path
from spectre.support.Machines import Machine, UnknownMachineError, this_machine


class TestMachines(unittest.TestCase):
    def setUp(self):
        self.machinefile_path = os.path.join(
            unit_test_build_path(), "TestMachine.yaml"
        )
        yaml.safe_dump(
            dict(
                Machine=dict(
                    Name="TestMachine",
                    Description="Just for testing",
                    DefaultProcsPerNode=15,
                    DefaultQueue="production",
                    DefaultTimeLimit="1-00:00:00",
                )
            ),
            open(self.machinefile_path, "w"),
        )

    def test_this_machine(self):
        with self.assertRaises(UnknownMachineError):
            this_machine("NonexistentMachinefile.yaml")
        machine = this_machine(self.machinefile_path)
        self.assertIsInstance(machine, Machine)
        self.assertEqual(machine.Name, "TestMachine")
        self.assertEqual(machine.Description, "Just for testing")
        self.assertEqual(machine.DefaultProcsPerNode, 15)
        self.assertEqual(machine.DefaultQueue, "production")
        self.assertEqual(machine.DefaultTimeLimit, "1-00:00:00")


if __name__ == "__main__":
    unittest.main(verbosity=2)
