# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre import Informer
try:
    # Fallback for Py2 that provides `assertRegex`
    import unittest2 as unittest
except:
    import unittest

VERSION_PATTERN = r'\d{4}\.\d{2}\.\d{2}(\.\d+)?'


class TestInformer(unittest.TestCase):
    def test_spectre_version(self):
        self.assertRegex(Informer.spectre_version(), VERSION_PATTERN)

    # The unit test path is unpredictable, but the last 12 characters must be
    # '/tests/Unit/'
    def test_unit_test_path(self):
        self.assertEqual(Informer.unit_test_path()[-12:], '/tests/Unit/')


if __name__ == '__main__':
    unittest.main(verbosity=2)
