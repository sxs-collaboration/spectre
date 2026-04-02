# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Fetch a GitHub issue and print it in a structured format for Claude.

If the issue is a random test failure (title contains 'random failure in'
and the body/comments contain MAKE_GENERATOR seed lines), a compact
[RANDOM_FAILURE] format is emitted instead of the full issue text.
"""

import argparse
import json
import re
import subprocess
import sys


def is_random_failure(title, all_text):
    """Return True if this looks like a MAKE_GENERATOR random failure issue."""
    if "random failure in" not in title.lower():
        return False
    return bool(re.search(r"Seed is:\s*\d+\s+from", all_text))


def extract_make_generator_seeds(all_text):
    """Parse 'Seed is: N from path:line' entries, returning {path:line: [seeds]}.

    CI logs often wrap long paths at '/' boundaries, so we strip
    timestamps and collapse continuation lines before matching.
    """
    # Strip CI log timestamps (e.g. "2023-09-11T19:44:29.1292895Z   ")
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*", "", all_text)
    # Collapse lines where a path wraps after a '/'
    text = re.sub(r"/\s*\n\s*", "/", text)
    pattern = r"Seed is:\s*(\d+)\s+from\s+(.*?\.[hct]pp:\d+)"
    matches = re.findall(pattern, text)

    grouped = {}
    for seed, raw_path in matches:
        # Canonicalize: keep only the part after tests/Unit/
        m = re.search(r"tests/Unit/(.*)", raw_path)
        canonical = m.group(1) if m else raw_path
        grouped.setdefault(canonical, [])
        if seed not in grouped[canonical]:
            grouped[canonical].append(seed)
    return grouped


def extract_test_name(title):
    """Extract the test name from a 'Random failure in TEST' title."""
    m = re.search(r"[Rr]andom\s+[Ff]ailure\s+in\s+(\S+)", title)
    return m.group(1) if m else title


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a GitHub issue via gh CLI"
    )
    parser.add_argument("issue_number", help="Issue number (e.g. 1727)")
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository in owner/repo form (default: current repo)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Always print full issue text, bypassing random-failure detection",
    )
    args = parser.parse_args()

    cmd = [
        "gh",
        "issue",
        "view",
        str(args.issue_number),
        "--json",
        (
            "number,title,state,author,assignees,"
            "createdAt,labels,url,body,comments"
        ),
    ]
    if args.repo:
        cmd.extend(["--repo", args.repo])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)

    title = data.get("title", "")
    body = data.get("body", "") or ""
    comments = data.get("comments", [])
    all_text = body + "\n" + "\n".join((c.get("body") or "") for c in comments)

    # --- Random failure detection ---
    if not args.full and is_random_failure(title, all_text):
        seeds = extract_make_generator_seeds(all_text)
        test_name = extract_test_name(title)
        print("[RANDOM_FAILURE]")
        print(f"Issue #{data['number']}: {title}")
        print(f"URL: {data.get('url', '')}")
        print(f"Test: {test_name}")
        print()
        print("Seeds by location (paths relative to tests/Unit/):")
        for path_line, seed_list in seeds.items():
            print(f"  {path_line} -> {', '.join(seed_list)}")
        sys.exit(0)

    # --- Normal issue output ---
    author = (data.get("author") or {}).get("login", "ghost")
    labels = ", ".join(lbl["name"] for lbl in data.get("labels", []))
    assignees = ", ".join(a["login"] for a in data.get("assignees", []))

    print(f"Issue #{data['number']}: {data['title']}")
    print(f"URL:     {data.get('url', '')}")
    print(f"State:   {data['state']}")
    print(f"Author:  {author}")
    print(f"Created: {data['createdAt']}")
    if assignees:
        print(f"Assigned: {assignees}")
    if labels:
        print(f"Labels:  {labels}")
    print()
    print((data.get("body") or "").strip())

    for comment in comments:
        comment_author = (comment.get("author") or {}).get("login", "ghost")
        print()
        print("---")
        print(f"Comment by: {comment_author} at {comment['createdAt']}")
        print((comment.get("body") or "").strip())


if __name__ == "__main__":
    main()
