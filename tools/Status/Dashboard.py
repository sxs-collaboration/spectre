# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Experimental dashboard for monitoring simulations.

This script provides a [Streamlit](https://streamlit.io) dashboard for
monitoring simulations. It uses the 'spectre.tools.Status' module to fetch
information about the jobs and then displays lots of plots and metrics that are
updated in real time.

To use this experimental feature, first make sure you have the following Python
packages installed in your environment in addition to the regular Python
dependencies:

```sh
pip install streamlit streamlit-autorefresh plotly
```

To spin up the dashboard, run the following command:

```sh
$BUILD_DIR/bin/python-spectre -m streamlit \
    run $SPECTRE_HOME/tools/Status/Dashboard.py
```

The dashboard will be available at the following URL by default:

- Dashboard: http://localhost:8501

You can forward the port through your SSH connection to open the dashboard in
your local browser. Note that VSCode forwards ports automatically and also
provides a you with a simple browser to view the dashboard within VSCode if you
prefer.

See the [Streamlit docs](https://docs.streamlit.io/develop/api-reference/cli/run)
for details.
"""

import logging
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from spectre.tools.Status.Status import (
    DEFAULT_COLUMNS,
    fetch_status,
    match_executable_status,
)
from spectre.Visualization.Plot import DEFAULT_MPL_STYLESHEET

logger = logging.getLogger(__name__)

plt.style.use(DEFAULT_MPL_STYLESHEET)

STATE_ICONS = {
    "RUNNING": ":material/arrow_forward:",
    "COMPLETED": ":material/check_circle:",
    "PENDING": ":material/pending:",
    "FAILED": ":material/block:",
    "TIMEOUT": ":material/hourglass_disabled:",
    "CANCELLED": ":material/cancel:",
}


def _render_page(job):
    st.header(job["JobName"])

    # Print run directory
    run_dir = Path(job["WorkDir"])
    st.write(run_dir)

    # Print README
    readme_files = [
        run_dir / "README.md",
    ]
    if job["SegmentsDir"]:
        readme_files += [
            Path(job["SegmentsDir"]) / "README.md",
            Path(job["SegmentsDir"]) / "../README.md",
        ]
    for readme_file in readme_files[::-1]:
        if readme_file.exists():
            st.markdown(readme_file.read_text())

    # Organize into tabs
    status_tab, input_file_tab, outfile_tab = st.tabs(
        ["Status", Path(job["InputFile"]).name, "spectre.out"]
    )

    # Print input file
    with input_file_tab:
        st.code(
            Path(job["InputFile"]).read_text(),
            language="yaml",
            line_numbers=True,
        )

    # Print outfile
    with outfile_tab:
        outfile = run_dir / "spectre.out"
        if outfile.exists():
            # Show only the last 50 lines because showing the full file can be
            # slow. We could show a button to load more lines.
            with open(outfile, "r") as open_outfile:
                st.code("".join(["...\n"] + open_outfile.readlines()[-50:]))
        else:
            st.write("No spectre.out file found.")

    # Render job status
    with status_tab:
        columns = list(DEFAULT_COLUMNS)
        columns.remove("User")
        columns.remove("JobName")
        st.table(pd.DataFrame([job[columns]]).set_index("JobID"))

        # Render status metrics
        executable_status = match_executable_status(job["ExecutableName"])
        with open(job["InputFile"], "r") as open_input_file:
            _, input_file = yaml.safe_load_all(open_input_file)
        status = executable_status.status(input_file, job["WorkDir"])
        for (field, unit), col in zip(
            executable_status.fields.items(),
            st.columns(len(executable_status.fields)),
        ):
            col.metric(
                (field + f" [{unit}]") if unit else field,
                (
                    executable_status.format(field, status[field])
                    if field in status
                    else "-"
                ),
            )

        # Executable-specific dashboard
        executable_status.render_dashboard(job, input_file)


# Fetch the job data
job_data = fetch_status(
    user=None,
    allusers=st.sidebar.toggle("Show all users", False),
    starttime=st.sidebar.text_input("Start time", "now-1day"),
)
if len(job_data) > 0:
    job_data.sort_values("JobID", inplace=True, ascending=False)
    job_data.dropna(subset=["WorkDir", "ExecutableName"], inplace=True)

# Refresh the page regularly
if st.sidebar.toggle("Auto-refresh", True):
    refresh_interval = st.sidebar.number_input(
        "Refresh interval [s]", min_value=1, value=30, format="%d"
    )
    count = st_autorefresh(interval=refresh_interval * 1000)

# Each job gets its own page. Jobs are grouped by user.
pages = {}
for username, user_data in job_data.groupby("User"):
    pages_user = []
    for _, job in user_data.iterrows():
        # Get a somewhat descriptive title for the page. This can be improved.
        # We probably want to make the job name more descriptive and show that
        # here.
        display_dir = (
            Path(job["SegmentsDir"])
            if job["SegmentsDir"]
            else Path(job["WorkDir"])
        )
        title = (
            job["JobName"]
            if job["JobName"] != job["ExecutableName"]
            else (display_dir.resolve().parent.name + " / " + display_dir.name)
        )
        pages_user.append(
            st.Page(
                partial(_render_page, job=job),
                title=f"{title} ({job['JobID']})",
                icon=STATE_ICONS.get(job["State"]),
                url_path=f"/{job['JobID']}",
            )
        )
    pages[username] = pages_user

# Set up navigation
if len(pages) > 0:
    page = st.navigation(pages)
    page.run()
else:
    st.write("No jobs found.")
