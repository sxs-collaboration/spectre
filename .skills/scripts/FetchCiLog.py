# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Fetch a failed GitHub Actions job log and extract failure context.

Retrieves the job summary and full log via the gh CLI, locates failure
lines, and prints a context window around each failure for the LLM.
"""

import argparse
import re
import subprocess
import sys

FAILURE_PATTERNS = [
    re.compile(r"FAILED:"),
    re.compile(r"\berror:"),
    re.compile(r"The following tests FAILED"),
    re.compile(r"##\[error\]"),
    re.compile(r"Process completed with exit code"),
    re.compile(r"\*\*\*Failed"),
]

# Patterns where only context *after* the failure line is useful
# (e.g. CTest's "***Failed" line is followed by the test output)
AFTER_ONLY_PATTERNS = {r"\*\*\*Failed"}

# Timestamp prefix on every GH Actions log line
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?")


def fetch_job_summary(job_id, owner, repo):
    """Fetch the job summary via 'gh run view --job'."""
    cmd = [
        "gh",
        "run",
        "view",
        "--job",
        str(job_id),
        "-R",
        f"{owner}/{repo}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"Warning: could not fetch job summary: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return None
    return result.stdout


def fetch_job_log(job_id, owner, repo):
    """Fetch the full job log via the GitHub API."""
    cmd = [
        "gh",
        "api",
        f"repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching log: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def strip_timestamps(lines):
    """Remove GH Actions timestamp prefixes from each line."""
    return [TIMESTAMP_RE.sub("", line) for line in lines]


def find_failure_lines(lines):
    """Return sorted list of (line_index, after_only) tuples.

    ``after_only`` is True when the matched pattern only needs context
    after the failure line (e.g. CTest's ``***Failed`` output).
    """
    hits = {}
    for i, line in enumerate(lines):
        for pattern in FAILURE_PATTERNS:
            if pattern.search(line):
                after_only = pattern.pattern in AFTER_ONLY_PATTERNS
                hits[i] = after_only
                break
    return hits


def merge_windows(hits, context, total_lines):
    """Merge overlapping [start, end] windows around each hit.

    ``hits`` is a dict {line_index: after_only}.  For after-only hits
    the window starts at the hit line itself (no before-context).
    """
    if not hits:
        return []
    windows = []
    for hit in sorted(hits):
        after_only = hits[hit]
        start = hit if after_only else max(0, hit - context)
        end = min(total_lines - 1, hit + context)
        if windows and start <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], end, windows[-1][2] | {hit})
        else:
            windows.append((start, end, {hit}))
    return windows


def print_window(lines, start, end, failure_indices):
    """Print a window of lines with line numbers, marking failures."""
    for i in range(start, end + 1):
        marker = ">>>" if i in failure_indices else "   "
        print(f"{marker} {i + 1:>6d} | {lines[i]}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a failed GitHub Actions job log and extract failure context."
        )
    )
    parser.add_argument("job_id", help="GitHub Actions job ID")
    parser.add_argument(
        "--owner",
        default="sxs-collaboration",
        help="Repository owner (default: sxs-collaboration)",
    )
    parser.add_argument(
        "--repo",
        default="spectre",
        help="Repository name (default: spectre)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=50,
        help="Number of lines before and after each failure (default: 50)",
    )
    args = parser.parse_args()

    # -- Job summary --
    summary = fetch_job_summary(args.job_id, args.owner, args.repo)
    if summary:
        print("=" * 60)
        print("JOB SUMMARY")
        print("=" * 60)
        print(summary.rstrip())
        print()

    # -- Fetch and process log --
    raw_log = fetch_job_log(args.job_id, args.owner, args.repo)
    raw_lines = raw_log.splitlines()
    lines = strip_timestamps(raw_lines)

    hits = find_failure_lines(lines)
    if not hits:
        print("No failure patterns found in the log.")
        print(f"Total log lines: {len(lines)}")
        sys.exit(1)

    windows = merge_windows(hits, args.context, len(lines))
    num_hits = len(hits)

    print("=" * 60)
    print(f"FAILURE LOG  (job {args.job_id}, {args.owner}/{args.repo})")
    print(
        f"  {num_hits} failure line(s) found, {len(windows)} context window(s)"
    )
    print(f"  Total log lines: {len(lines)}")
    print("=" * 60)

    for idx, (start, end, failure_indices) in enumerate(windows):
        print()
        print(
            f"--- Window {idx + 1}/{len(windows)} "
            f"(lines {start + 1}-{end + 1}) ---"
        )
        print_window(lines, start, end, failure_indices)

    print()
    print("=" * 60)
    print("END OF FAILURE CONTEXT")
    print("=" * 60)


if __name__ == "__main__":
    main()
