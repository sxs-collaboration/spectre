# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.tools.CharmSimplifyTraces import extract_first_template_parameter


class TestCharmSimplifyTraces(unittest.TestCase):
    def test_extract_first_template_parameter(self):
        # Test the example in the docs
        self.assertEqual(
            extract_first_template_parameter(
                "A<1, B<2, 4>, 3>, C, D, E<7, 8, F<9>>"
            ),
            "A<1, B<2, 4>, 3>",
        )
        # This test should be expanded further!


if __name__ == "__main__":
    unittest.main(verbosity=2)
