# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import TranslationMapOptions


class TestTranslationMap(unittest.TestCase):
    def test_construction(self):
        translation_map = TranslationMapOptions(
            [[1.0, -1.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        self.assertNotEqual(translation_map, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
