#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import contextlib
import io
import os
import unittest

import h5py
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.Profile import profile, profile_command


def parse_table(output):
    """Parse a printed table into its header and a list of (row name, values)
    pairs, in the order printed."""
    lines = output.strip().split("\n")
    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        # Row names may contain spaces (e.g. "No block"), so split from the
        # right.
        name, *values = line.rsplit(maxsplit=len(header))
        rows.append((name.strip(), [float(value) for value in values]))
    return header, rows


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(), "IO/H5/Python"
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.h5_filename = os.path.join(self.test_dir, "TestProfile.h5")
        if os.path.exists(self.h5_filename):
            os.remove(self.h5_filename)

        # Exclusive times are chosen so that Proc0 has a total of 400, Proc1
        # has a total of 600, and the overall total is 1000.
        #
        # Proc0/Act/Block0: count 4, exclusive 200 (two categories, rows out
        #                   of order)
        # Proc0/Act/Block1: count 2, exclusive 100
        # Proc0/Other:      count 1, exclusive 96
        # Proc0/Tiny:       count 7, exclusive 4
        # Proc1/Act/Block1: count 6, exclusive 300
        # Proc1/Act/Block2/Block0: count 5, exclusive 100 (first block wins)
        # Proc1/Other:      count 1, exclusive 200 (first column is named
        #                   "Step count" and must not be counted)
        dat_files = {
            "Proc0/Act/Block0": (
                [
                    "Time",
                    "Continue count",
                    "Continue exclusive",
                    "Continue inclusive",
                    "Halt count",
                    "Halt exclusive",
                    "Halt inclusive",
                ],
                # The latest row (Time = 2) is not the last row.
                [
                    [0.0, 1.0, 10.0, 20.0, 0.0, 0.0, 0.0],
                    [2.0, 3.0, 150.0, 400.0, 1.0, 50.0, 60.0],
                    [1.0, 2.0, 100.0, 300.0, 1.0, 40.0, 50.0],
                ],
            ),
            "Proc0/Act/Block1": (
                ["Time", "count", "exclusive", "inclusive"],
                [[1.0, 1.0, 30.0, 30.0], [2.0, 2.0, 100.0, 900.0]],
            ),
            "Proc0/Other": (
                ["Time", "count", "exclusive", "inclusive"],
                [[2.0, 1.0, 96.0, 96.0]],
            ),
            "Proc0/Tiny": (
                ["Time", "count", "exclusive", "inclusive"],
                [[2.0, 7.0, 4.0, 4.0]],
            ),
            "Proc1/Act/Block1": (
                ["Time", "count", "exclusive", "inclusive"],
                [[2.0, 6.0, 300.0, 300.0]],
            ),
            "Proc1/Act/Block2/Block0": (
                ["Time", "count", "exclusive", "inclusive"],
                [[2.0, 5.0, 100.0, 100.0]],
            ),
            "Proc1/Other": (
                ["Step count", "count", "exclusive", "inclusive"],
                [[1000.0, 1.0, 200.0, 200.0]],
            ),
        }
        with spectre_h5.H5File(self.h5_filename, "a") as h5file:
            for path, (legend, data) in dat_files.items():
                dat_file = h5file.insert_dat(
                    "/Profiling/" + path, legend=legend, version=0
                )
                dat_file.append(data)
                h5file.close_current_object()

        # Groups that are not valid top-level profiling groups
        with h5py.File(self.h5_filename, "a") as h5file:
            h5file.create_group("Empty")
            h5file.create_dataset("NotAGroup", data=[1.0])

    def tearDown(self):
        if os.path.exists(self.h5_filename):
            os.remove(self.h5_filename)

    def run_profile(self, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            profile(self.h5_filename, "Profiling", **kwargs)
        return parse_table(output.getvalue())

    def assert_table(self, actual, expected_header, expected_rows):
        header, rows = actual
        self.assertEqual(header, expected_header)
        self.assertEqual(
            [name for name, _ in rows], [name for name, _ in expected_rows]
        )
        for (name, values), (_, expected_values) in zip(rows, expected_rows):
            self.assertEqual(len(values), len(expected_values), name)
            for value, expected_value in zip(values, expected_values):
                # The table is printed with 3 significant figures.
                self.assertAlmostEqual(
                    value,
                    expected_value,
                    delta=1e-3 * abs(expected_value),
                    msg=name,
                )

    def test_actions(self):
        header = ["cum(%)", "exc(%)", "count"]
        self.assert_table(
            self.run_profile(),
            header,
            [("Act", [70.0, 70.0, 17.0]), ("Other", [99.6, 29.6, 2.0])],
        )
        self.assert_table(
            self.run_profile(cutoff=0.0),
            header,
            [
                ("Act", [70.0, 70.0, 17.0]),
                ("Other", [99.6, 29.6, 2.0]),
                ("Tiny", [100.0, 0.4, 7.0]),
            ],
        )
        self.assert_table(
            self.run_profile(cutoff=50.0),
            header,
            [("Act", [70.0, 70.0, 17.0])],
        )
        self.assert_table(
            self.run_profile(proc=1),
            header,
            [("Act", [66.7, 66.7, 11.0]), ("Other", [100.0, 33.3, 1.0])],
        )

    def test_blocks(self):
        self.assert_table(
            self.run_profile(blocks=True, cutoff=50.0),
            ["exc(%)"],
            [
                ("Block1", [40.0]),
                ("No block", [30.0]),
                ("Block0", [20.0]),
                ("Block2", [10.0]),
            ],
        )
        self.assert_table(
            self.run_profile(blocks=True, proc=1),
            ["exc(%)"],
            [("Block1", [50.0]), ("No block", [33.3]), ("Block2", [16.7])],
        )

    def test_balance(self):
        header = ["exc(%)", "count"]
        self.assert_table(
            self.run_profile(balance="Act", cutoff=90.0),
            header,
            [("Proc0", [75.0, 6.0]), ("Proc1", [66.7, 11.0])],
        )
        # Processes without matching entries are still listed
        self.assert_table(
            self.run_profile(balance="Block2"),
            header,
            [("Proc1", [16.7, 5.0]), ("Proc0", [0.0, 0.0])],
        )
        # A trailing .dat is ignored when matching, and the label can be
        # given with or without it
        for label in ["Other", "Other.dat"]:
            self.assert_table(
                self.run_profile(balance=label),
                header,
                [("Proc1", [33.3, 1.0]), ("Proc0", [24.0, 1.0])],
            )

    def test_errors(self):
        def assert_error(message, group="Profiling", **kwargs):
            with self.assertRaisesRegex(ValueError, message):
                profile(self.h5_filename, group, **kwargs)

        assert_error(
            r"Unable to find group Nope\. Top-level groups are:\n"
            r"\['Empty', 'Profiling'\]",
            group="Nope",
        )
        assert_error("NotAGroup is a dataset", group="NotAGroup")
        assert_error("Empty .* It is empty", group="Empty")
        assert_error(
            r"Profiling/Proc0 does not look like a top-level profiling group"
            r".* It contains other entries: Act, Other\.dat, Tiny\.dat\.",
            group="Profiling/Proc0",
        )
        assert_error("Unable to find group Proc2 in Profiling", proc=2)
        assert_error("'--blocks' and '--balance'", blocks=True, balance="Act")
        assert_error("'--proc' and '--balance'", proc=0, balance="Act")
        assert_error(
            "No profiling entries match the label 'Nope'", balance="Nope"
        )

    def test_cli(self):
        runner = CliRunner()
        for args, kwargs in [
            ([], {}),
            (["--cutoff", "0"], {"cutoff": 0.0}),
            (["--blocks", "--proc", "1"], {"blocks": True, "proc": 1}),
            (["--balance", "Act"], {"balance": "Act"}),
        ]:
            result = runner.invoke(
                profile_command,
                [self.h5_filename, "Profiling", *args],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0, result.output)
            expected = io.StringIO()
            with contextlib.redirect_stdout(expected):
                profile(self.h5_filename, "Profiling", **kwargs)
            self.assertEqual(result.output, expected.getvalue())

        result = runner.invoke(
            profile_command,
            [self.h5_filename, "Profiling", "--proc", "-1"],
        )
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
