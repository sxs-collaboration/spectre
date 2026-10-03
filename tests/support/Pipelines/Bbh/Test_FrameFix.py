# Distributed under the MIT License.
# See LICENSE.txt for details.

import shutil
import unittest
from pathlib import Path

import click
import h5py
import numpy as np
import scri

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import available_subfiles
from spectre.Pipelines.Bbh.FrameFix import (
    _spectre_waveforms,
    _write_cce_file,
    frame_fix,
    frame_fix_command,
)
from spectre.Visualization.PlotCce import plot_cce

# Keep this small so the iterative BMS solve stays fast. Real CCE output uses
# an observation l_max of 8.
L_MAX = 4
EXTRACTION_RADIUS = 200.0
SUBFILE_NAME = f"SpectreR{int(EXTRACTION_RADIUS):04d}.cce"
QUANTITIES = [
    "EthInertialRetardedTime",
    "News",
    "Psi0",
    "Psi1",
    "Psi2",
    "Psi3",
    "Psi4",
    "Strain",
]


def _mode_index(ell, m):
    """Index of the (ell, m) mode in m-varies-fastest ordering from ell = 0."""
    return ell**2 + ell + m


def write_test_cce_file(file_name: Path, num_times: int = 150):
    """Write a crude inspiral-like waveform in the SpECTRE Cce format.

    Only modes with ell >= |spin weight| are populated, since the others vanish
    identically for spin-weighted quantities.
    """
    num_modes = (L_MAX + 1) ** 2
    times = np.linspace(0.0, 3000.0, num_times)
    phase = 0.02 * (1.0 + times / times[-1]) * times
    amplitude = 0.1 * (1.0 + times / times[-1])
    strain = np.zeros((num_times, num_modes), dtype=complex)
    strain[:, _mode_index(2, 2)] = amplitude * np.exp(1j * phase)
    strain[:, _mode_index(2, -2)] = amplitude * np.exp(-1j * phase)
    # A constant offset in the memory mode, which the frame fixing acts on
    strain[:, _mode_index(2, 0)] = 0.01
    news = np.gradient(strain, times[1] - times[0], axis=0)
    psi2 = np.zeros((num_times, num_modes), dtype=complex)
    # Psi2 has spin weight zero, and its (0, 0) mode is minus the Bondi mass
    psi2[:, _mode_index(0, 0)] = -1.0
    data = {
        "Strain": strain,
        "News": news,
        "Psi4": np.gradient(news, times[1] - times[0], axis=0),
        "Psi3": 0.01 * news,
        "Psi2": psi2,
        "Psi1": 0.001 * strain,
        "Psi0": 0.001 * strain,
        "EthInertialRetardedTime": np.zeros_like(strain),
    }
    with spectre_h5.H5File(file_name=str(file_name), mode="a") as h5file:
        cce_file = h5file.insert_cce(path=SUBFILE_NAME, l_max=L_MAX, version=1)
        for i in range(num_times):
            row = {}
            for name, modes in data.items():
                values = np.empty(1 + 2 * num_modes)
                # CCE writes the time at the extraction radius, not the
                # retarded time
                values[0] = times[i] + EXTRACTION_RADIUS
                values[1::2] = modes[i].real
                values[2::2] = modes[i].imag
                row[name] = values
            cce_file.append(row)


class TestFrameFix(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/FrameFix"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.cce_reduction_file = (
            self.test_dir / "CharacteristicExtractReduction.h5"
        )
        write_test_cce_file(self.cce_reduction_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_conventions(self):
        # Loading with scri and writing back out without any BMS transformation
        # must reproduce the input, which tests the conversion between scri's
        # Moreschi-Boyle conventions and SpECTRE's conventions.
        abd = scri.SpEC.file_io.create_abd_from_h5(
            file_format="SpECTRECCE_v1", file_name=str(self.cce_reduction_file)
        )
        output_file = self.test_dir / "RoundTrip.h5"
        _write_cce_file(
            output_file=output_file,
            subfile_name=SUBFILE_NAME,
            times=abd.t,
            waveforms=_spectre_waveforms(abd),
            extraction_radius=EXTRACTION_RADIUS,
        )
        with (
            h5py.File(self.cce_reduction_file, "r") as original,
            h5py.File(output_file, "r") as roundtrip,
        ):
            original_data = original[SUBFILE_NAME]
            roundtrip_data = roundtrip[SUBFILE_NAME]
            self.assertEqual(
                list(original_data["Strain"].attrs["Legend"]),
                list(roundtrip_data["Strain"].attrs["Legend"]),
            )
            for name in ["Strain", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4"]:
                np.testing.assert_allclose(
                    roundtrip_data[name][()],
                    original_data[name][()],
                    atol=1e-14,
                    err_msg=f"'{name}' does not round-trip through scri",
                )
            # The News is recomputed as a spectral time derivative of the
            # strain, so it only agrees with the finite-difference News in the
            # test data to the accuracy of that derivative
            np.testing.assert_allclose(
                roundtrip_data["News"][()],
                original_data["News"][()],
                atol=1e-2,
            )

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        frame_fix_command.main(
            args=[
                str(self.cce_reduction_file),
                "--t-0-superrest",
                "1000",
                "--padding-time",
                "200",
            ],
            standalone_mode=False,
        )
        output_file = (
            self.test_dir / "CharacteristicExtractReductionFrameFixed.h5"
        )
        self.assertTrue(output_file.exists())

        with h5py.File(output_file, "r") as open_h5_file:
            self.assertEqual(
                available_subfiles(open_h5_file, extension=".cce"),
                [SUBFILE_NAME],
            )
            frame_fixed = open_h5_file[SUBFILE_NAME]
            self.assertEqual(sorted(frame_fixed.keys()), QUANTITIES)
            # The diagnostic that the frame fixing supersedes is NaN, not zero
            diagnostic = frame_fixed["EthInertialRetardedTime"][()]
            self.assertTrue(np.isfinite(diagnostic[:, 0]).all())
            self.assertTrue(np.isnan(diagnostic[:, 1:]).all())
            self.assertEqual(
                list(frame_fixed["Strain"].attrs["Legend"][:3]),
                ["time", "Real Y_0,0", "Imag Y_0,0"],
            )
            strain = frame_fixed["Strain"][()]
            self.assertEqual(strain.shape[1], 1 + 2 * (L_MAX + 1) ** 2)
            self.assertTrue(np.isfinite(strain).all())
            # Times are written in the same convention as the raw CCE output,
            # i.e. offset by the extraction radius
            self.assertGreater(strain[0, 0], EXTRACTION_RADIUS)

        # The output is in the same format as the raw CCE output, so it can be
        # plotted with the same tooling
        plot_cce(str(output_file), modes=["Real Y_2,2", "Imag Y_2,2"])

        # Refuse to overwrite the output unless '--force' is given
        with self.assertRaises(click.UsageError):
            frame_fix(self.cce_reduction_file, t_0_superrest=1000.0)

    def test_frame_fixing_window(self):
        with self.assertRaises(click.UsageError):
            frame_fix(self.cce_reduction_file, t_0_superrest=1.0e10)
        # scri uses a wider window than 'padding_time', and silently makes do
        # with less data, so point that out. This t_0 is inside the data, so it
        # passes the check above, but the wider window is not.
        with self.assertLogs(
            "spectre.Pipelines.Bbh.FrameFix", level="WARNING"
        ) as logs:
            frame_fix(
                self.cce_reduction_file,
                output_file=self.test_dir / "NearTheEnd.h5",
                t_0_superrest=2900.0,
                padding_time=200.0,
            )
        self.assertIn("shorter window than intended", logs.output[0])

    def test_refuses_to_overwrite_input(self):
        # Writing the output over the input would destroy the CCE data
        with self.assertRaisesRegex(
            click.UsageError, "CCE output that we read"
        ):
            frame_fix(
                self.cce_reduction_file,
                output_file=self.cce_reduction_file,
                force=True,
            )
        self.assertTrue(self.cce_reduction_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
