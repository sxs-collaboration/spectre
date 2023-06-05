# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import unittest

import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.tools.Status.ExecutableStatus import match_executable_status
from spectre.tools.Status.Status import get_executable_name, get_input_file


class TestExecutableStatus(unittest.TestCase):
    def setUp(self):
        self.input_file = {
            "Observers": {"ReductionFileName": "Reductions"},
            "EventsAndTriggers": [
                {
                    "Trigger": "Always",
                    "Events": [
                        {"ObserveTimeStep": {"SubfileName": "TimeSteps"}}
                    ],
                }
            ],
        }
        self.work_dir = os.path.join(
            unit_test_build_path(), "tools/ExecutableStatus"
        )
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        with spectre_h5.H5File(
            os.path.join(self.work_dir, "Reductions.h5"), "w"
        ) as open_h5_file:
            # Time data
            time_subfile = open_h5_file.insert_dat(
                "/TimeSteps",
                legend=[
                    "Time",
                    "Slab size",
                    "Minimum Walltime",
                    "Maximum Walltime",
                ],
                version=0,
            )
            time_subfile.append([0.0, 0.5, 0.5, 1.0])
            time_subfile.append([1.0, 0.5, 2.0, 3.0])
            time_subfile.append([2.0, 0.5, 3.0, 4.0])
            open_h5_file.close_current_object()
            # Control systems data
            rotation_subfile = open_h5_file.insert_dat(
                "/ControlSystems/Rotation/z",
                legend=["FunctionOfTime"],
                version=0,
            )
            rotation_subfile.append([0.0])
            rotation_subfile.append([np.pi])
            open_h5_file.close_current_object()
            # AH data
            aha_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhA_Centers",
                legend=[
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],
                version=0,
            )
            aha_subfile.append([1.0, 0.0, 0.0])
            open_h5_file.close_current_object()
            ahb_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhB_Centers",
                legend=[
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],
                version=0,
            )
            ahb_subfile.append([-1.0, 0.0, 0.0])
            open_h5_file.close_current_object()
            # Constraints
            constraints_subfile = open_h5_file.insert_dat(
                "/Norms", legend=["L2Norm(ConstraintEnergy)"], version=0
            )
            constraints_subfile.append([1.0e-3])

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_evolution_status(self):
        executable_status = match_executable_status("EvolveSomething")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status, {"Time": 2.0, "Speed": 2640.0})
        self.assertEqual(executable_status.format("Time", 1.5), "1.5")
        self.assertEqual(executable_status.format("Speed", 1.2), "1.2")

    def test_evolve_bbh_status(self):
        executable_status = match_executable_status("EvolveGhBinaryBlackHole")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status["Time"], 2.0)
        self.assertEqual(status["Speed"], 2640.0)
        self.assertEqual(status["Orbits"], 0.5)
        self.assertEqual(status["Separation"], 2.0)
        self.assertEqual(status["Constraint Energy"], 1.0e-3)

    def test_evolve_single_bh_status(self):
        executable_status = match_executable_status("EvolveGhSingleBlackHole")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status["Time"], 2.0)
        self.assertEqual(status["Speed"], 2640.0)
        self.assertEqual(status["Constraint Energy"], 1.0e-3)


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.slurm_comment = (
            "SPECTRE_INPUT_FILE=path/to/input/file\n"
            "SPECTRE_EXECUTABLE=path/to/executable\n"
        )
        self.work_dir = os.path.join(unit_test_build_path(), "tools/Status")
        self.input_file_path = os.path.join(self.work_dir, "InputFile.yaml")
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.input_file_path, "w") as open_input_file:
            yaml.safe_dump_all([{"Executable": "MyExec"}, {}], open_input_file)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_get_input_file(self):
        self.assertEqual(
            get_input_file(self.slurm_comment, self.work_dir),
            os.path.join(self.work_dir, "path/to/input/file"),
        )
        self.assertIsNone(
            get_input_file("", os.path.join(self.work_dir, "nonexistent"))
        )
        self.assertEqual(
            get_input_file("", self.work_dir), self.input_file_path
        )

    def test_get_executable_name(self):
        self.assertEqual(
            get_executable_name(self.slurm_comment, self.input_file_path),
            "executable",
        )
        self.assertIsNone(get_executable_name("", None))
        self.assertEqual(
            get_executable_name("", self.input_file_path), "MyExec"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
