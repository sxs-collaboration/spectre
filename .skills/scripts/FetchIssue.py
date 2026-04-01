# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Fetch a GitHub issue and print it in a structured format for Claude."""

import argparse
import json
import subprocess
import sys


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

    for comment in data.get("comments", []):
        comment_author = (comment.get("author") or {}).get("login", "ghost")
        print()
        print("---")
        print(f"Comment by: {comment_author} at {comment['createdAt']}")
        print((comment.get("body") or "").strip())


if __name__ == "__main__":
    main()
