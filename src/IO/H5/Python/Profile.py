#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re

import click
import h5py
import numpy as np
import pandas as pd


def extract_timings(group, name_transform):
    """Total the call counts and exclusive times of the Dat files in `group`.

    Each Dat file is assigned a name by calling `name_transform` with the
    components of its path relative to `group`. Returns a DataFrame indexed by
    those names, with columns "count" and "exclusive".
    """
    rows = []

    def add_row(path, item):
        if not isinstance(item, h5py.Dataset):
            return
        # Rows hold cumulative values but are not necessarily in order, so use
        # the row with the largest value in the first (ordering) column.
        data = item[()]
        latest = data[np.argmax(data[:, 0])]
        row = {
            "name": name_transform(path.split("/")),
            "count": 0.0,
            "exclusive": 0.0,
        }
        for label, value in zip(item.attrs["Legend"][1:], latest[1:]):
            for column in ("count", "exclusive"):
                if label.endswith(column):
                    row[column] += value
        rows.append(row)

    group.visititems(add_row)
    return (
        pd.DataFrame(rows, columns=["name", "count", "exclusive"])
        .groupby("name")
        .sum()
        .rename_axis(None)
    )


def top_name(path):
    return path[0].removesuffix(".dat")


def block_name(path):
    """Return the first path component of the form "Block[digits]", ignoring
    a trailing .dat extension, or "No block" if there is none."""
    for component in path:
        component = component.removesuffix(".dat")
        if re.fullmatch(r"Block\d+", component):
            return component
    return "No block"


def get_proc_groups(h5file, profiling_group, proc):
    """Return a dict mapping 'ProcN' names to their H5 groups under
    `profiling_group`, restricted to 'Proc{proc}' if `proc` is not None."""
    group = h5file.get(profiling_group)
    if group is None:
        groups = [
            name
            for name, item in h5file.items()
            if isinstance(item, h5py.Group)
        ]
        raise ValueError(
            f"Unable to find group {profiling_group}. Top-level "
            f"groups are:\n{groups}"
        )

    # The profiling group should contain only per-process subgroups. A common
    # mistake is to pass one of those subgroups instead.
    if not isinstance(group, h5py.Group):
        raise ValueError(
            f"{profiling_group} is a dataset, not a group. Pass the "
            "top-level profiling group, which contains one 'ProcN' "
            "subgroup per process."
        )
    unexpected = [
        name
        for name, item in group.items()
        if not (isinstance(item, h5py.Group) and re.fullmatch(r"Proc\d+", name))
    ]
    if unexpected or len(group) == 0:
        if unexpected:
            shown = ", ".join(unexpected[:5])
            if len(unexpected) > 5:
                shown += f" (and {len(unexpected) - 5} more)"
            contents = f"It contains other entries: {shown}."
        else:
            contents = "It is empty."
        raise ValueError(
            f"{profiling_group} does not look like a top-level profiling "
            f"group, which should contain only 'ProcN' subgroups. {contents} "
            "To summarize a single process, pass the top-level group and use "
            "'--proc N'."
        )

    if proc is None:
        return dict(group.items())
    name = f"Proc{proc}"
    if name not in group:
        raise ValueError(
            f"Unable to find group {name} in {profiling_group}. "
            f"Known groups are:\n{list(group.keys())}"
        )
    return {name: group[name]}


def print_table(table, columns):
    """Print the `columns` of `table`, where `columns` maps column names to
    headers."""
    print(
        table[list(columns)].to_string(
            header=list(columns.values()), float_format="%.3g"
        )
    )


def profile(
    filename,
    profiling_group,
    cutoff=1.0,
    blocks=False,
    proc=None,
    balance=None,
):
    """Summarize profiling data from a SpECTRE HDF5 file

    Read the profiling data under PROFILING_GROUP, the full path to the
    top-level H5 group of profiling data, and print a table of where the time
    was spent, sorted by exclusive time. The latest row of each Dat file is
    used, and timings are summed over all processes, or taken only from the
    'ProcN' subgroup if '--proc N' is given.

    By default, timings are summarized by the top-level entry in each process's
    group. The table shows the cumulative exclusive time and the exclusive time
    of each entry, both as percentages of the total exclusive time, followed by
    the number of calls. Entries whose exclusive time is below the '--cutoff'
    percentage are omitted.

    With '--blocks', timings are instead summarized by block. An entry is
    assigned to the first component of its path of the form 'Block' followed by
    digits, or to 'No block' if there is none.

    With '--balance LABEL', the table instead shows how the time spent in
    LABEL is distributed over processes. For each process, the exclusive time
    and count of every entry with LABEL as any component of its path under
    'ProcN' (ignoring a trailing .dat extension) are summed. The exclusive
    time is shown as a percentage of that process's total exclusive time, and
    the rows are sorted by that percentage. '--cutoff' is ignored, and
    '--blocks' and '--proc' cannot be used with '--balance'.
    """
    if balance is not None:
        if blocks:
            raise ValueError(
                "'--blocks' and '--balance' cannot be used together."
            )
        if proc is not None:
            raise ValueError(
                "'--proc' and '--balance' cannot be used together."
            )

        # Group each process's entries by whether they match the label.
        def name_transform(path):
            return any(
                balance in (component, component.removesuffix(".dat"))
                for component in path
            )

    elif blocks:
        name_transform = block_name
    else:
        name_transform = top_name

    with h5py.File(filename, "r") as h5file:
        timings = {
            name: extract_timings(group, name_transform)
            for name, group in get_proc_groups(
                h5file, profiling_group, proc
            ).items()
        }

    if balance is not None:
        print_balance(timings, balance)
        return

    # Ties are sorted alphabetically by name.
    table = pd.concat(timings.values()).groupby(level=0).sum()
    table = table.sort_values(by="exclusive", ascending=False, kind="stable")
    table["exclusive"] *= 100.0 / table["exclusive"].sum()
    table["cumulative"] = table["exclusive"].cumsum()

    if blocks:
        print_table(table, {"exclusive": "exc(%)"})
    else:
        print_table(
            table[table["exclusive"] >= cutoff],
            {"cumulative": "cum(%)", "exclusive": "exc(%)", "count": "count"},
        )


def print_balance(timings, label):
    """Print the share of each process's time spent in entries matching
    `label`. `timings` maps process names to the result of `extract_timings`
    with entries keyed by whether they match."""
    if not any(True in timing.index for timing in timings.values()):
        raise ValueError(f"No profiling entries match the label '{label}'.")

    rows = {}
    for name, timing in timings.items():
        matched = timing.reindex([True], fill_value=0.0).iloc[0]
        total_time = timing["exclusive"].sum()
        rows[name] = {
            "exclusive": (
                100.0 * matched["exclusive"] / total_time if total_time else 0.0
            ),
            "count": matched["count"],
        }

    # Put processes in numerical order first so that ties stay in that order.
    table = pd.DataFrame.from_dict(rows, orient="index")
    table = table.loc[
        sorted(table.index, key=lambda name: int(name.removeprefix("Proc")))
    ]
    table = table.sort_values(by="exclusive", ascending=False, kind="stable")
    print_table(table, {"exclusive": "exc(%)", "count": "count"})


@click.command(name="profile", help=profile.__doc__)
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument("profiling_group")
@click.option(
    "--cutoff",
    type=float,
    default=1.0,
    show_default=True,
    help="Omit entries with exclusive time below this percentage.",
)
@click.option(
    "--blocks",
    is_flag=True,
    help="Summarize by block. Ignores --cutoff.",
)
@click.option(
    "--proc",
    type=click.IntRange(min=0),
    metavar="N",
    help="Only use data from the ProcN group.  [default: all processes]",
)
@click.option(
    "--balance",
    metavar="LABEL",
    help=(
        "Show the percentage of each process's time spent in entries with "
        "LABEL in their path. Ignores --cutoff. Cannot be combined with "
        "--blocks or --proc."
    ),
)
def profile_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    profile(**kwargs)


if __name__ == "__main__":
    profile_command(help_option_names=["-h", "--help"])
