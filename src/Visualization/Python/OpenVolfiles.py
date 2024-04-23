# Distributed under the MIT License.
# See LICENSE.txt for details.

import fnmatch
import functools
import logging
from typing import Iterable, Optional

import click
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.Visualization.ReadH5 import list_observations, select_observation


def open_volfiles(
    h5_files: Iterable[str], subfile_name: str, obs_id: Optional[int] = None
):
    """Opens each volume data file in turn

    Arguments:
      h5_files: List of H5 files containing volume data
      subfile_name: Name of the H5 volume data subfile
      obs_id: Optional. If specified, only yield volume data files that
        contain the specified observation ID.
    """
    import spectre.IO.H5 as spectre_h5

    for h5_file in h5_files:
        with spectre_h5.H5File(h5_file, "r") as open_h5_file:
            volfile = open_h5_file.get_vol(subfile_name)
            if obs_id is None or obs_id in volfile.list_observation_ids():
                yield volfile


def parse_step(ctx, param, value):
    """Parse a CLI option as integer. Accepts 'first' and 'last'."""
    if value is None:
        return None
    if value.lower() == "first":
        return 0
    if value.lower() == "last":
        return -1
    return int(value)


def parse_point(ctx, param, value):
    """Parse a CLI option as comma-separated list of floats."""
    if not value:
        return None
    return np.array(list(map(float, value.split(","))))


def parse_points(ctx, param, values):
    """Parse a CLI option as multiple comma-separated list of floats."""
    if not values:
        return None
    points = [parse_point(ctx, param, value) for value in values]
    dim = len(points[0])
    if any([len(point) != dim for point in points]):
        raise click.BadParameter(
            "All specified points must have the same dimension"
        )
    return np.array(points)


def open_volfiles_command(obs_id_required=False, multiple_vars=False):
    """CLI options for accessing volume data files

    Use this decorator to add options for accessing volume data to a CLI
    command. The decorated function should accept the following arguments in
    addition to any other arguments it needs:

    - h5_files: List of paths to h5 files containing volume data.
    - subfile_name: Name of subfile within h5 file containing volume data.
    - obs_id: Selected observation ID. Can be None if obs_id_required is False.
    - obs_time: Time of observation. None if obs_id is also None.
    - vars or var_name: If 'multiple_vars' is True, the list of selected
        variable names in the volume data. Otherwise, the single selected
        variable name.
    """

    def decorator(f):
        @click.argument(
            "h5_files",
            nargs=-1,
            type=click.Path(
                exists=True, file_okay=True, dir_okay=False, readable=True
            ),
        )
        @click.option(
            "--subfile-name",
            "-d",
            help=(
                "Name of subfile within h5 file containing volume data to plot."
            ),
        )
        @click.option(
            "--list-vars",
            "-l",
            is_flag=True,
            help="Print available variables and exit.",
        )
        @click.option(
            "--var",
            "-y",
            "vars_patterns",
            multiple=multiple_vars,
            help=(
                "Variable to plot. List any tensor components "
                "in the volume data file, such as 'Shift_x'. "
                "Also accepts glob patterns like 'Shift_*'."
            )
            + (" Can be specified multiple times." if multiple_vars else ""),
        )
        @click.option(
            "--list-observations",
            "--list-times",
            "list_times",
            is_flag=True,
            help="Print all available observation times and exit.",
        )
        @click.option(
            "--step",
            callback=parse_step,
            help=(
                "Observation step number. Specify '-1' or 'last' "
                "for the last step in the file. "
                "Mutually exclusive with '--time'."
            ),
        )
        @click.option(
            "--time",
            type=float,
            help=(
                "Observation time. "
                "The observation step closest to the specified "
                "time is selected. "
                "Mutually exclusive with '--step'."
            ),
        )
        # Preserve the original function's name and docstring
        @functools.wraps(f)
        def command(
            h5_files,
            subfile_name,
            list_vars,
            vars_patterns,
            list_times,
            step,
            time,
            **kwargs,
        ):
            # Script should be a noop if input files are empty
            if not h5_files:
                return

            # Print available subfile names and exit
            if not subfile_name:
                import rich.columns

                with spectre_h5.H5File(h5_files[0], "r") as open_h5_file:
                    available_subfiles = open_h5_file.all_vol_files()
                if len(available_subfiles) == 1:
                    subfile_name = available_subfiles[0]
                else:
                    rich.print(rich.columns.Columns(available_subfiles))
                    return

            # Print available observations/times and exit
            if list_times:
                import rich.columns

                all_obs_ids, all_obs_times = list_observations(
                    open_volfiles(h5_files, subfile_name)
                )
                rich.print(
                    rich.columns.Columns(f"{time:g}" for time in all_obs_times)
                )
                return
            # Select observation
            if step is None and time is None:
                if obs_id_required:
                    all_obs_ids, all_obs_times = list_observations(
                        open_volfiles(h5_files, subfile_name)
                    )
                    if len(all_obs_ids) == 1:
                        obs_id, obs_time = all_obs_ids[0], all_obs_times[0]
                    else:
                        raise click.UsageError(
                            "Specify '--step' or '--time' to select an"
                            " observation in the volume data. The volume data"
                            f" contains {len(all_obs_ids)} observations between"
                            f" times {all_obs_times[0]} and"
                            f" {all_obs_times[-1]}."
                        )
                else:
                    obs_id, obs_time = None, None
            else:
                obs_id, obs_time = select_observation(
                    open_volfiles(h5_files, subfile_name), step=step, time=time
                )

            # Print available variables and exit
            for volfile in open_volfiles(h5_files, subfile_name, obs_id):
                all_vars = volfile.list_tensor_components(
                    obs_id or volfile.list_observation_ids()[0]
                )
                break
            if list_vars or not vars_patterns:
                import rich.columns

                rich.print(rich.columns.Columns(all_vars))
                return
            # Expand globs in vars
            vars = []
            if not multiple_vars:
                vars_patterns = [vars_patterns]
            for var_pattern in vars_patterns:
                matched_vars = fnmatch.filter(all_vars, var_pattern)
                if not matched_vars:
                    raise click.UsageError(
                        f"The pattern '{var_pattern}' matches no variables. "
                        f"Available variables are: {all_vars}"
                    )
                vars.extend(matched_vars)
            # Remove duplicates. Ordering is lost, but that's not important here.
            vars = list(set(vars))
            if not multiple_vars and len(vars) > 1:
                raise click.UsageError("Only one variable can be selected.")

            return f(
                h5_files=h5_files,
                subfile_name=subfile_name,
                obs_id=obs_id,
                obs_time=obs_time,
                **(
                    dict(vars=vars) if multiple_vars else dict(var_name=vars[0])
                ),
                **kwargs,
            )

        return command

    return decorator
