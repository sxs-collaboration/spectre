# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from click.testing import CliRunner

from spectre.Domain import ElementId
from spectre.Visualization.ElementIdToParaview import (
    element_id_to_paraview_command,
    element_id_to_short_id,
)


class TestElementIdToParaview(unittest.TestCase):
    def test_element_id_to_short_id(self):
        eid_str = "[B2,(L2I3,L1I0,L1I1)]"
        expected = ElementId[3](eid_str).to_short_id()
        self.assertEqual(element_id_to_short_id(eid_str), expected)

    def test_auto_detects_dimension(self):
        self.assertEqual(
            element_id_to_short_id("[B0,(L1I0)]"),
            ElementId[1]("[B0,(L1I0)]").to_short_id(),
        )
        self.assertEqual(
            element_id_to_short_id("[B0,(L1I0,L2I3)]"),
            ElementId[2]("[B0,(L1I0,L2I3)]").to_short_id(),
        )

    def test_cli_single(self):
        runner = CliRunner()
        result = runner.invoke(
            element_id_to_paraview_command,
            ["[B2,(L2I3,L1I0,L1I1)]"],
        )
        self.assertEqual(result.exit_code, 0)
        expected = str(ElementId[3]("[B2,(L2I3,L1I0,L1I1)]").to_short_id())
        self.assertEqual(result.output.strip(), expected)

    def test_cli_multiple(self):
        runner = CliRunner()
        ids = [
            "[B0,(L2I3,L1I0,L1I1)]",
            "[B2,(L2I3,L1I0,L1I1)]",
        ]
        result = runner.invoke(element_id_to_paraview_command, ids)
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            int(lines[0]),
            ElementId[3]("[B0,(L2I3,L1I0,L1I1)]").to_short_id(),
        )
        self.assertEqual(
            int(lines[1]),
            ElementId[3]("[B2,(L2I3,L1I0,L1I1)]").to_short_id(),
        )

    def test_cli_mixed_dimensions(self):
        runner = CliRunner()
        ids = ["[B0,(L1I0)]", "[B2,(L2I3,L1I0,L1I1)]"]
        result = runner.invoke(element_id_to_paraview_command, ids)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("same dimension", result.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
