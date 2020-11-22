# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import glob
import h5py
import logging
import numpy as np
import numpy.testing as npt
import os
import yaml

# For Py2 compatibility
try:
    import unittest2 as unittest
except:
    import unittest


class H5Check:
    """Describes a particular comparison between H5 datasets or groups

    When passed a `unittest.testcase` object to the `perform_checks`,
    the described checks are performed on the H5 files in the appointed run
    directory

    Attributes:
        unit_test: The `unittest.testcase` object, used to invoke asserts
        test_h5_label: An identifier string for the test
        h5_glob: The shell glob matching the h5 files to test
        test_h5_entity: The h5 path for the group or dataset to check
        expected_h5_entity: The h5 path for the expected group or dataset
        absolute_tolerance: The absolute tolerance for approximation checks
        relative_tolerance: The relative tolerance for approximation checks
    """
    def __init__(self, unit_test, test_h5_label, h5_glob, test_h5_entity,
                 expected_h5_entity, absolute_tolerance, relative_tolerance,
                 skip_columns):
        """Initializer for H5Check

        Note: the `unit_test` argument must be the unit test object -- this
        class is to be constructed within a test case, and must have access
        to the calling unit test object so that it can access the assert
        and subTest member functions
        """
        self.unit_test = unit_test
        self.h5_glob = h5_glob
        self.test_h5_label = test_h5_label
        self.test_h5_entity = test_h5_entity
        self.expected_h5_entity = expected_h5_entity
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = (0.0 if relative_tolerance is None else
                                   float(relative_tolerance))
        self.skip_columns = ([] if skip_columns is None else skip_columns)

    def check_h5_file(self, h5_file, test_entity, expected_entity):
        """Perform the unit test comparisons between the dataset or group

        If `expected_entity` is `None`, then the `test_entity` is compared to
        0.0 instead (using only the absolute tolerance)

        Args:
            h5_file: An h5 File object on which to perform checks
            test_entity: The h5 path string to the test dataset or group
            expected_entity: The h5 path to the expected dataset or group
        """

        if isinstance(h5_file[test_entity], h5py.Dataset):
            logging.info("Checking dataset : " + test_entity)
            with self.unit_test.subTest(test_entity=test_entity,
                                        expected_entity=expected_entity):
                test_data = h5_file[test_entity][()]
                column_mask = [
                    x not in self.skip_columns
                    for x in range(test_data.shape[1])
                ]
                if self.expected_h5_entity is not None:
                    expected_data = h5_file[expected_entity][()]
                    self.unit_test.assertEqual(
                        test_data.shape, expected_data.shape,
                        "test and expected h5 datasets must have identical"
                        " sizes.")
                    self.unit_test.assertEqual(
                        test_data.dtype, expected_data.dtype,
                        "test and expected h5 datasets must have identical"
                        " types.")
                    if (test_data.dtype == np.dtype(np.float) or
                            test_data.dtype == np.dtype(np.complexfloating)):
                        npt.assert_allclose(test_data[:, column_mask],
                                            expected_data[:, column_mask],
                                            rtol=self.relative_tolerance,
                                            atol=self.absolute_tolerance)
                    else:
                        self.unit_test.assertEqual(test_data, expected_data)
                else:
                    if (test_data.dtype == np.dtype(np.float) or
                            test_data.dtype == np.dtype(np.complexfloating)):
                        npt.assert_allclose(test_data[:, column_mask],
                                            0.0,
                                            rtol=self.relative_tolerance,
                                            atol=self.absolute_tolerance)
                    else:
                        self.assertTrue(
                            False,
                            msg="cannot test non-numeric data without an"
                            " expected data set to compare against")
        elif isinstance(h5_file[test_entity], h5py.Group):
            test_keys = set(h5_file[self.test_entity].keys())
            checks_passed = True
            if self.expected_h5_entity is not None:
                expected_keys = set(h5_file[self.expected_entity].keys())
                keys_difference = test_keys ^ expected_keys
                self.unit_test.assertEqual(
                    keys_difference, {},
                    "test and expected h5 groups must have identical" +
                    "subgroups and data members. Found differences: " +
                    str(keys_difference) + "\nin comparing group " +
                    test_entity + " to group " + expected_entity)
                for key in test_keys:
                    checks_passed = checks_passed and self.check_h5_file(
                        h5_file, test_entity + "/" + key,
                        expected_entity + "/" + key)
            else:
                for key in test_keys:
                    checks_passed = checks_passed and self.check_h5_file(
                        h5_file, test_entity + "/" + key, None)
            return checks_passed

    def perform_check(self, run_directory):
        """Apply the h5 check to every h5 file within the `run_directory`
        that matches the glob `self.h5_glob`.

        If neither the expected nor test objects are present in an h5 file,
        that file is skipped. However, if none of the matching h5 files have
        the requested datasets, the test fails, under the assumption that
        failure to produce any of the anticipated data should be regarded
        as an error in the executable.
        """
        checks_performed = False
        logging.info("Performing checks: " +
                     os.path.join(run_directory, self.h5_glob))
        for filename in glob.glob(os.path.join(run_directory, self.h5_glob)):
            logging.info("Checking file: " + filename)
            with self.unit_test.subTest(filename=filename):
                with h5py.File(filename, 'r') as check_h5:
                    if self.test_h5_entity in check_h5 or (
                            self.expected_h5_entity is not None
                            and self.expected_h5_entity in check_h5):
                        checks_performed = True
                        self.unit_test.assertTrue(
                            self.test_h5_entity in check_h5)
                        self.unit_test.assertTrue(
                            self.expected_h5_entity is None
                            or self.expected_h5_entity in check_h5)
                        self.check_h5_file(check_h5, self.test_h5_entity,
                                           self.expected_h5_entity)
        self.unit_test.assertTrue(
            checks_performed, "No checks performed for glob: " +
            os.path.join(run_directory, self.h5_glob))


def read_h5_checks_config_lines(input_filename):
    """Parse the OutputFileChecks lines into a yaml string
    """
    with open(input_filename, 'r') as input_file:
        lines = input_file.readlines()
        found_config = False
        for i, line in enumerate(lines):
            if line.strip() == '# OutputFileChecks:':
                found_config = True
                yield line[2:]
                continue
            if found_config:
                if line.startswith('#   '):
                    yield line[2:]
                else:
                    return
        raise RuntimeError(
            "Could not find '# OutputFileChecks:' in input file. Please"
            " specify the h5 fields to check in the test yaml."
            " See tools/CheckOutputFiles.py for syntax details.")


class H5CheckTestCase(unittest.TestCase):
    """The unit test object for performing all H5 checks for a given input file

    The parameters for the H5 output checks are determined by the
    command-line arguments. The first argument (sys.argv[1]) is the yaml
    input file. The second argument is the directory in which to find the
    run's.h5 files.

    The H5 checks and arguments are parsed from the yaml file comments
    we require a format like:
    ```
    # OutputFileChecks:
    #   - Label: "label"
    #     Subfile: "/h5_name.dat"
    #     FileGlob: "VolumeData*.h5"
    #     AbsoluteTolerance: 1e-12
    #   - Label: "another_label"
    #     Subfile: "/h5_group_name"
    #     FileGlob: "ReductionData*.h5"
    #     ExpectedDataSubfile: "/expected_h5_group_name"
    #     AbsoluteTolerance: 1e-11
    #     RelativeTolerance: 1e-6
    #     SkipColumns: [0, 1]
    ```
    In particular, the entire comment block starting from `# OutputFileChecks:`
    and ending where the indentation or comment breaks must be parsable as yaml.
    `SkipColumns` must be a list in brackets of the indices of columns to omit
    from the test.
    """
    def test_h5_output(self):
        h5_check_list = []
        config_lines = read_h5_checks_config_lines(self.input_filename)
        parsed_yaml = yaml.safe_load(''.join(config_lines))
        for check_block in parsed_yaml['OutputFileChecks']:
            logging.info("Parsed File check : " + check_block.get("Label"))
            h5_check_list.append(
                H5Check(self, check_block.get("Label"),
                        check_block.get("FileGlob"),
                        check_block.get("Subfile"),
                        check_block.get("ExpectedDataSubfile"),
                        check_block.get("AbsoluteTolerance"),
                        check_block.get("RelativeTolerance"),
                        check_block.get("SkipColumns")))
        for h5_check in h5_check_list:
            with self.subTest(test=h5_check.test_h5_label):
                h5_check.perform_check(self.run_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filename')
    parser.add_argument('--run-directory')
    logging.basicConfig(level=logging.INFO)
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=H5CheckTestCase)
    del duplicate_test_case
    # Use of full command-line arguments breaks the unit-test framework
    # (which needs to take its own command-line arguments), so we only pass
    # on the remaining args after we've retrieved the ones used by
    # `H5CheckTestCase`.
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
