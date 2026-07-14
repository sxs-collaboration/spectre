# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest
from pathlib import Path

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.Cce import run_cce, run_cce_command


class TestCce(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Cce"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()

        # Set up directories to hold input and output files
        self.inspiral_dir = self.test_dir / "Inspiral"
        self.ringdown_dir = self.test_dir / "Ringdown"
        self.inspiral_seg_dir = self.inspiral_dir / "Segment_0000"
        self.ringdown_seg_dir = self.ringdown_dir / "Segment_0000"
        if os.path.exists(self.inspiral_dir):
            shutil.rmtree(self.inspiral_dir)
        if os.path.exists(self.ringdown_dir):
            shutil.rmtree(self.ringdown_dir)
        os.makedirs(self.inspiral_dir, exist_ok=True)
        os.makedirs(self.ringdown_dir, exist_ok=True)
        os.makedirs(self.inspiral_seg_dir, exist_ok=True)
        os.makedirs(self.ringdown_seg_dir, exist_ok=True)

        self.inspiral_input_file_path = os.path.join(
            self.inspiral_seg_dir, "BondiSachsCceR0200.h5"
        )
        self.ringdown_input_file_path = os.path.join(
            self.ringdown_seg_dir, "BondiSachsCceR0200.h5"
        )
        self.bad_filename = os.path.join(
            self.inspiral_seg_dir, "BadFileName.h5"
        )
        self.different_radius_file = os.path.join(
            self.inspiral_seg_dir, "BondiSachsCceR0100.h5"
        )
        with open(self.bad_filename, "w") as file:
            file.write("This file is not in the correct format.\n")
        with open(self.different_radius_file, "w") as file:
            file.write(
                "This file has a different radius in its filename than"
                " the others.\n"
            )

        # Make BondiSachs data for CCE
        self.wave_inspiral_1 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(0, 10.0, 0.1)]
        )
        self.wave_inspiral_2 = np.array(
            [[t, 2 * np.sin(t), 2 * np.cos(t)] for t in np.arange(0, 10.0, 0.1)]
        )
        self.wave_ringdown_1 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(9, 14.0, 0.1)]
        )
        self.wave_ringdown_2 = np.array(
            [[t, 2 * np.sin(t), 2 * np.cos(t)] for t in np.arange(9, 14.0, 0.1)]
        )

        # Generate 2 h5 files, one for each segment, with two dat files each
        with spectre_h5.H5File(
            file_name=self.inspiral_input_file_path, mode="r+"
        ) as h5file:
            beta_datfile = h5file.insert_dat(
                path="/Beta", legend=["Time", "Re(0,0)", "Re(1,0)"], version=0
            )
            beta_datfile.append(self.wave_inspiral_1)
        with spectre_h5.H5File(
            file_name=self.inspiral_input_file_path, mode="r+"
        ) as h5file:
            drj_datfile = h5file.insert_dat(
                path="/DrJ",
                legend=["Time", "Re(0,0)", "Re(1,0)"],
                version=0,
            )
            drj_datfile.append(self.wave_inspiral_2)
        with spectre_h5.H5File(
            file_name=self.ringdown_input_file_path, mode="r+"
        ) as h5file:
            beta_datfile = h5file.insert_dat(
                path="/Beta", legend=["Time", "Re(0,0)", "Re(1,0)"], version=0
            )
            beta_datfile.append(self.wave_ringdown_1)
        with spectre_h5.H5File(
            file_name=self.ringdown_input_file_path, mode="r+"
        ) as h5file:
            drj_datfile = h5file.insert_dat(
                path="/DrJ",
                legend=["Time", "Re(0,0)", "Re(1,0)"],
                version=0,
            )
            drj_datfile.append(self.wave_ringdown_2)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        with self.assertRaises(ValueError) as context:
            run_cce_command.main(
                args=[
                    str(self.bad_filename),
                    "--run-dir",
                    str(self.test_dir / "01_Output"),
                    "--executable",
                    str(self.bin_dir / "CharacteristicExtract"),
                    "--no-submit",
                ],
                standalone_mode=False,
            )
            str(context.exception),
            (
                "The provided BondiSachs file does not end with 'RXXXX.h5'."
                " Modify the filename to include the extraction radius in the"
                " format 'NameOfFileRXXXX.h5'. For example, if the extraction"
                " radius is 200, the filename should end with 'R0200.h5'."
            ),
        with self.assertRaises(ValueError) as context:
            run_cce_command.main(
                args=[
                    str(self.inspiral_input_file_path),
                    str(self.ringdown_input_file_path),
                    str(self.different_radius_file),
                    "--run-dir",
                    str(self.test_dir / "01_Output"),
                    "--executable",
                    str(self.bin_dir / "CharacteristicExtract"),
                    "--no-submit",
                ],
                standalone_mode=False,
            )
            str(context.exception),
            (
                "Contradicting extraction radii for files specified. Ensure all"
                " BondiSachs files end with the same extraction radius."
            ),
        run_cce_command.main(
            args=[
                str(self.inspiral_seg_dir / "BondiSachsCceR0200.h5"),
                str(self.ringdown_seg_dir / "BondiSachsCceR0200.h5"),
                "--run-dir",
                str(self.test_dir / "01_Output"),
                "--executable",
                str(self.bin_dir / "CharacteristicExtract"),
                "--no-submit",
            ],
            standalone_mode=False,
        )
        self.assertTrue((self.test_dir / "01_Output/Cce.yaml").exists())
        self.assertTrue(
            (self.test_dir / "01_Output/combinedBondiSachsCceR0200.h5").exists()
        )
        run_cce_command.main(
            args=[
                self.inspiral_input_file_path,
                "--run-dir",
                str(self.test_dir / "02_Output"),
                "-E",
                str(self.bin_dir / "CharacteristicExtract"),
                "--no-submit",
            ],
            standalone_mode=False,
        )
        self.assertTrue((self.test_dir / "02_Output/Cce.yaml").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
