# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
from pathlib import Path
from typing import Optional, Union

import click
import h5py
import numpy as np
import scri

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5 import available_subfiles

logger = logging.getLogger(__name__)

# scri requires this format for the CCE subfile name
SUBFILE_PATTERN = re.compile(r"SpectreR(\d{4})\.cce$")

# Extra time on either side of the padding window that scri slices out of the
# data to determine the BMS transformation, see
# 'scri.asymptotic_bondi_data.map_to_superrest_frame'
SCRI_EXTRA_PADDING_TIME = 200.0


def _resolve_subfile(cce_reduction_file: Path):
    """Find the Cce subfile to transform and its extraction radius.

    Note that scri picks the subfile itself when it reads the file, so we can't
    offer a choice here: it takes the first one whose name contains "Spectre".
    """
    with h5py.File(cce_reduction_file, "r") as open_h5_file:
        cce_subfiles = available_subfiles(open_h5_file, extension=".cce")
    if not cce_subfiles:
        raise click.UsageError(
            f"Could not find any Cce subfiles in H5 file {cce_reduction_file}."
            " Cce subfiles must end with the extension '.cce'."
        )
    if len(cce_subfiles) > 1:
        raise click.UsageError(
            f"The H5 file {cce_reduction_file} has {len(cce_subfiles)} Cce"
            f" subfiles ({', '.join(cce_subfiles)}), but scri only"
            " supports one extraction radius per file. Split them into"
            " separate files first."
        )
    subfile_name = cce_subfiles[0]
    match = re.search(SUBFILE_PATTERN, subfile_name)
    if not match:
        raise click.UsageError(
            f"The Cce subfile '{subfile_name}' is not named 'SpectreRXXXX.cce',"
            " so the extraction radius can't be determined. scri requires this"
            " format."
        )
    return subfile_name, float(match.group(1))


def _spectre_waveforms(abd):
    """Waveform quantities in SpECTRE's conventions vs. scri's conventions.

    Internally scri uses the Moreschi-Boyle conventions, so undo the conversion
    that 'scri.SpEC.file_io.create_abd_from_h5' applied when it loaded the data
    with 'convention="SpEC"'. The factors here are the inverse of scri's
    'conversion_factor["spec"] = [2, -sqrt(2), 1, -1/sqrt(2), 0.5, 0.5]' for
    '[Psi0, Psi1, Psi2, Psi3, Psi4, h]'.
    """
    # 'abd.sigma' holds the conjugate of half the strain, so this is the strain
    # in SpECTRE's conventions (the same as scri's 'abd.h', but keeping the
    # l = 0, 1 modes that SpECTRE always writes out).
    sigma_bar = abd.sigma.bar
    return {
        "Strain": 2.0 * sigma_bar,
        # SpECTRE's News is the first time derivative of the strain
        "News": 2.0 * sigma_bar.dot,
        "Psi0": abd.psi0 / 2.0,
        "Psi1": abd.psi1 / (-np.sqrt(2.0)),
        "Psi2": abd.psi2,
        "Psi3": -np.sqrt(2.0) * abd.psi3,
        "Psi4": 2.0 * abd.psi4,
    }


def _write_cce_file(
    output_file: Path,
    subfile_name: str,
    times,
    waveforms: dict,
    extraction_radius,
):
    """Write frame-fixed waveforms in SpECTRE's Cce format.

    The layout matches the output of the CCE executable, so tools like
    'spectre plot cce' work on the result.
    """
    num_modes = next(iter(waveforms.values())).shape[1]
    l_max = int(round(np.sqrt(num_modes))) - 1
    # scri works in retarded time, so it subtracts the extraction radius when it
    # reads a Cce subfile. Add it back so the time column of a Cce subfile
    # always means the same thing and the data round-trips through scri.
    time_column = np.asarray(times) + extraction_radius
    # Interleave real and imaginary parts of the modes, matching the legend that
    # 'h5::Cce' generates: 'time, Real Y_0,0, Imag Y_0,0, Real Y_1,-1, ...'
    rows = {}
    for name, modes in waveforms.items():
        modes = np.asarray(modes)
        row = np.empty((len(time_column), 1 + 2 * num_modes))
        row[:, 0] = time_column
        row[:, 1::2] = modes.real
        row[:, 2::2] = modes.imag
        rows[name] = row
    # CCE writes "EthInertialRetardedTime" alongside the waveform quantities to
    # diagnose the coordinate transformation it performed at future null
    # infinity. The frame fixing supersedes that transformation, so we have
    # nothing meaningful to put here, but the Cce file format requires the
    # dataset. Fill it with NaN.
    rows["EthInertialRetardedTime"] = np.full(
        (len(time_column), 1 + 2 * num_modes), np.nan
    )
    rows["EthInertialRetardedTime"][:, 0] = time_column

    with spectre_h5.H5File(file_name=str(output_file), mode="a") as h5file:
        cce_file = h5file.insert_cce(path=subfile_name, l_max=l_max, version=1)
        for i in range(len(time_column)):
            cce_file.append({name: row[i] for name, row in rows.items()})


def frame_fix(
    cce_reduction_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    t_0_superrest: Optional[float] = None,
    padding_time: float = 200.0,
    junk_time: float = 500.0,
    ch_mass: Optional[float] = None,
    superrest_dt: Optional[float] = 5.0,
    force: bool = False,
) -> Path:
    """Transform CCE output into the correct BMS frame.

    Waveforms that the CCE executable writes at future null infinity are in the
    wrong Bondi-Metzner-Sachs (BMS) frame, which shows up as an offset in the
    strain that doesn't decay during ringdown. This maps the data to the
    superrest frame using scri, following
    https://scri.readthedocs.io/en/latest/tutorial_abd.html

    The result is written in the same format as the raw CCE output, so it can be
    plotted with 'spectre plot cce'. Note that 'EthInertialRetardedTime' is a
    diagnostic of the transformation that CCE performed, which the frame fixing
    supersedes, so it is filled with NaN in the output.

    See arXiv:2405.08868 for a review of BMS transformations and gravitational
    memory, and cite scri (https://github.com/moble/scri) if you use this.

    \f
    Arguments:
      cce_reduction_file: The CCE reduction file to transform, e.g.
        'CharacteristicExtractReduction.h5'.
      output_file: Where to write the frame-fixed data. Defaults to the input
        file name with 'FrameFixed' appended, next to the input file.
      t_0_superrest: Retarded time at which to map to the superrest frame.
        Defaults to 'junk_time' plus 'padding_time' after the start of the data.
      padding_time: Length of the time window around 't_0_superrest' used to
        determine the BMS transformation. A few hundred M, or a couple of
        orbits, is usually sufficient.
      junk_time: How long the CCE junk radiation lasts. Only used to choose a
        default for 't_0_superrest'.
      ch_mass: Total Christodoulou mass of the system, used to make the
        waveforms dimensionless. Defaults to no rescaling, which is what you
        want when the simulation already has unit total mass (as the BBH
        pipeline sets up).
      superrest_dt: Time spacing of the data used to determine the BMS
        transformation. The transformation is applied to the full data, so this
        only trades accuracy of the transformation against runtime, not the
        resolution of the output. Pass None to use the full data.
      force: Overwrite the output file if it already exists.
    """
    cce_reduction_file = Path(cce_reduction_file).resolve()
    if output_file is None:
        output_file = cce_reduction_file.with_name(
            cce_reduction_file.stem + "FrameFixed" + cce_reduction_file.suffix
        )
    output_file = Path(output_file)
    if output_file.exists():
        if output_file.samefile(cce_reduction_file):
            raise click.UsageError(
                f"The output file {output_file} is the CCE output that we read"
                " from. Write the frame-fixed data to a different file, so the"
                " raw CCE output is preserved."
            )
        if not force:
            raise click.UsageError(
                f"The output file {output_file} already exists. Use '--force' /"
                " '-f' to overwrite it."
            )
        output_file.unlink()

    subfile_name, extraction_radius = _resolve_subfile(cce_reduction_file)

    # scri works in retarded time, i.e. it subtracts the extraction radius from
    # the time column of the CCE output, so choose the transformation times in
    # that convention as well
    with h5py.File(cce_reduction_file, "r") as open_h5_file:
        times = open_h5_file[subfile_name]["Strain"][:, 0] - extraction_radius

    if t_0_superrest is None:
        # The BMS transformation must be determined from a window that is free
        # of junk radiation, so skip 'junk_time' plus the padding.
        # The junk radiation time is a crude heuristic. It should be replaced by
        # a relaxation time determined from the data.
        t_0_superrest = times[0] + junk_time + padding_time
    if t_0_superrest - padding_time < times[0] or t_0_superrest > times[-1]:
        raise click.UsageError(
            f"The frame-fixing window [{t_0_superrest - padding_time},"
            f" {t_0_superrest}] is not contained in the data, which covers"
            f" [{times[0]}, {times[-1]}] in retarded time. Adjust"
            " '--t-0-superrest' and '--padding-time'."
        )
    logger.info(
        f"Mapping to the superrest frame at t = {t_0_superrest:g} over a window"
        f" of {padding_time:g}, both in retarded time (the data covers"
        f" [{times[0]:g}, {times[-1]:g}]). Make sure this window is free of"
        " junk radiation."
    )
    # scri determines the transformation from a window that is wider than
    # 'padding_time' (see 'map_to_superrest_frame'), and silently uses whatever
    # data it finds there, so point out when that's less than it asked for
    scri_window = padding_time + SCRI_EXTRA_PADDING_TIME
    if (
        t_0_superrest - scri_window < times[0]
        or t_0_superrest + scri_window > times[-1]
    ):
        logger.warning(
            f"scri determines the transformation from t = {t_0_superrest:g} +/-"
            f" {scri_window:g}, but the data only covers [{times[0]:g},"
            f" {times[-1]:g}]. It will use the data it has, so the"
            " transformation is determined from a shorter window than intended."
        )

    abd = scri.SpEC.file_io.create_abd_from_h5(
        file_format="SpECTRECCE_v1",
        file_name=str(cce_reduction_file),
        ch_mass=ch_mass,
    )

    # Determine the BMS transformation from data that is sampled coarsely in
    # time, then apply it to the full data. CCE output is often sampled far more
    # densely than the iterative solve needs, and this keeps the solve from
    # taking hours without throwing away any of the output. See
    # https://scri.readthedocs.io/en/latest/tutorial_abd.html
    if superrest_dt is not None:
        abd_for_bms = abd.interpolate(
            np.arange(abd.t[0], abd.t[-1], superrest_dt)
        )
    else:
        abd_for_bms = abd
    _, bms_transformation, _ = abd_for_bms.map_to_superrest_frame(
        t_0=t_0_superrest, padding_time=padding_time
    )
    logger.info(f"BMS transformation: {bms_transformation}")
    abd = abd.transform(
        supertranslation=bms_transformation.supertranslation,
        frame_rotation=bms_transformation.frame_rotation.components,
        boost_velocity=bms_transformation.boost_velocity,
    )

    _write_cce_file(
        output_file=output_file,
        subfile_name=subfile_name,
        times=abd.t,
        waveforms=_spectre_waveforms(abd),
        extraction_radius=extraction_radius,
    )
    logger.info(f"Frame-fixed waveforms written to {output_file}")
    return output_file


@click.command(name="frame-fix", help=frame_fix.__doc__)
@click.argument(
    "cce_reduction_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Where to write the frame-fixed data. Defaults to the input file name"
        " with 'FrameFixed' appended."
    ),
)
@click.option(
    "--t-0-superrest",
    type=float,
    help=(
        "Retarded time at which to map to the superrest frame."
        " [default: 'junk-time' plus 'padding-time' after the start of the"
        " data]"
    ),
)
@click.option(
    "--padding-time",
    type=float,
    default=200.0,
    show_default=True,
    help=(
        "Length of the time window around 't-0-superrest' used to determine the"
        " BMS transformation."
    ),
)
@click.option(
    "--junk-time",
    type=float,
    default=500.0,
    show_default=True,
    help=(
        "How long the CCE junk radiation lasts. Only used to choose a default"
        " for 't-0-superrest'."
    ),
)
@click.option(
    "--ch-mass",
    type=float,
    help=(
        "Total Christodoulou mass of the system, used to make the waveforms"
        " dimensionless. [default: no rescaling]"
    ),
)
@click.option(
    "--superrest-dt",
    type=float,
    default=5.0,
    show_default=True,
    help=(
        "Time spacing of the data used to determine the BMS transformation. The"
        " transformation is applied to the full data, so this doesn't change"
        " the resolution of the output."
    ),
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
def frame_fix_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    frame_fix(**kwargs)


if __name__ == "__main__":
    frame_fix_command(help_option_names=["-h", "--help"])
