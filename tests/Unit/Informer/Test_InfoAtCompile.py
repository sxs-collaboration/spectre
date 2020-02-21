# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre import Informer as Info
import unittest


class TestInformer(unittest.TestCase):
    def test_spectre_version(self):
        num_ver = str(Info.spectre_major_version()) + "." + \
            str(Info.spectre_minor_version()) + "." + \
            str(Info.spectre_patch_version())
        self.assertEqual(num_ver, Info.spectre_version())

    # The unit test path is unpredictable, but the last 12 characters must be
    # '/tests/Unit/'
    def test_unit_test_path(self):
        self.assertEqual(Info.unit_test_path()[-12:], '/tests/Unit/')


if __name__ == '__main__':
    unittest.main(verbosity=2)
