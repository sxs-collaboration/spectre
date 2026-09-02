# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path
from spectre.support.BinDirectory import BIN_DIR_NAME, BinDirectory
from spectre.support.Logging import configure_logging


class TestBinDirectory(unittest.TestCase):
    def test_this(self):
        self.assertEqual(
            BinDirectory.this().path,
            Path(unit_test_build_path(), "../..", BIN_DIR_NAME).resolve(),
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
