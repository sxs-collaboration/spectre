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
        with open(self.machinefile_path, "w") as open_machinefile:
            yaml.safe_dump(
                dict(
                    Machine=dict(
                        Name="TestMachine",
                        Description="Just for testing",
                        DefaultProcsPerNode=15,
                        DefaultQueue="production",
                        DefaultTimeLimit="1-00:00:00",
                        LaunchCommandSingleNode=["mpirun", "-n", "1"],
                    )
                ),
                open_machinefile,
            )

    def test_this_machine(self):
        with self.assertRaises(UnknownMachineError):
            this_machine("NonexistentMachinefile.yaml")
        self.assertIsNone(
            this_machine("NonexistentMachinefile.yaml", raise_exception=False)
        )
        machine = this_machine(self.machinefile_path)
        self.assertIsInstance(machine, Machine)
        self.assertEqual(machine.Name, "TestMachine")
        self.assertEqual(machine.Description, "Just for testing")
        self.assertEqual(machine.DefaultProcsPerNode, 15)
        self.assertEqual(machine.DefaultQueue, "production")
        self.assertEqual(machine.DefaultTimeLimit, "1-00:00:00")
        self.assertEqual(machine.LaunchCommandSingleNode, ["mpirun", "-n", "1"])
        self.assertEqual(
            machine.launch_command,
            ["mpirun", "-n", "1"] if machine.on_compute_node() else [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
