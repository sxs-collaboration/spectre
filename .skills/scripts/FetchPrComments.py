# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Fetch PR review data and print it in a structured format for Claude."""

import argparse
import json
import subprocess
import sys


def run_gh_command(cmd):
    """Run a gh CLI command and return parsed JSON, or exit on error."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def fetch_pr_metadata(pr_number, repo=None):
    """Fetch PR metadata, comments, reviews, and files via gh pr view."""
    cmd = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--json",
        (
            "number,title,state,author,baseRefName,headRefName,"
            "body,comments,reviews,files,url,createdAt,reviewDecision"
        ),
    ]
    if repo:
        cmd.extend(["--repo", repo])
    return run_gh_command(cmd)


def fetch_review_threads(pr_number, repo=None):
    """Fetch review threads with resolution status via GraphQL API."""
    if repo:
        owner, name = repo.split("/", 1)
    else:
        detect_cmd = ["gh", "repo", "view", "--json", "owner,name"]
        repo_data = run_gh_command(detect_cmd)
        owner = repo_data["owner"]["login"]
        name = repo_data["name"]

    query = """query {
      repository(owner:"%s", name:"%s") {
        pullRequest(number:%d) {
          reviewThreads(first:100) {
            totalCount
            pageInfo { hasNextPage }
            nodes {
              isResolved
              isOutdated
              path
              line
              originalLine
              startLine
              originalStartLine
              subjectType
              comments(first:50) {
                nodes {
                  body
                  author { login }
                  path
                  diffHunk
                  line
                  originalLine
                  startLine
                  originalStartLine
                  createdAt
                }
              }
            }
          }
        }
      }
    }""" % (owner, name, int(pr_number))

    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    data = run_gh_command(cmd)

    threads_data = (
        data.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("reviewThreads", {})
    )

    if threads_data.get("pageInfo", {}).get("hasNextPage"):
        print(
            (
                "WARNING: PR has more than 100 review threads. "
                "Only the first 100 are shown."
            ),
            file=sys.stderr,
        )

    return threads_data.get("nodes", [])


def print_pr_header(data):
    """Print PR metadata header."""
    author = (data.get("author") or {}).get("login", "ghost")
    print(f"PR #{data['number']}: {data['title']}")
    print(f"URL:             {data.get('url', '')}")
    print(f"State:           {data['state']}")
    print(f"Author:          {author}")
    print(f"Created:         {data['createdAt']}")
    print(f"Base <- Head:    {data['baseRefName']} <- {data['headRefName']}")
    print(f"Review decision: {data.get('reviewDecision', 'NONE')}")


def print_changed_files(files):
    """Print changed files list with addition/deletion counts."""
    print()
    print("=" * 60)
    print("CHANGED FILES")
    print("=" * 60)
    for f in files:
        print(f"  +{f['additions']:-4d} -{f['deletions']:-4d}  {f['path']}")
    total_add = sum(f["additions"] for f in files)
    total_del = sum(f["deletions"] for f in files)
    print(f"  {'':>10}  ({len(files)} files, +{total_add} -{total_del} total)")


def print_pr_body(body):
    """Print PR description."""
    body = (body or "").strip()
    if body:
        print()
        print("=" * 60)
        print("PR DESCRIPTION")
        print("=" * 60)
        print(body)


def print_top_level_comments(comments):
    """Print top-level PR conversation comments."""
    if not comments:
        return
    print()
    print("=" * 60)
    print(f"TOP-LEVEL COMMENTS ({len(comments)})")
    print("=" * 60)
    for c in comments:
        author = (c.get("author") or {}).get("login", "ghost")
        print()
        print(f"--- Comment by {author} at {c['createdAt']} ---")
        print((c.get("body") or "").strip())


def print_review_summaries(reviews):
    """Print review summaries (APPROVED, CHANGES_REQUESTED, etc.)."""
    summaries = [r for r in reviews if (r.get("body") or "").strip()]
    if not summaries:
        return
    print()
    print("=" * 60)
    print(f"REVIEW SUMMARIES ({len(summaries)})")
    print("=" * 60)
    for r in summaries:
        author = (r.get("author") or {}).get("login", "ghost")
        state = r.get("state", "COMMENTED")
        print()
        print(f"--- [{state}] Review by {author} at {r['submittedAt']} ---")
        print((r.get("body") or "").strip())


def print_review_threads(threads):
    """Print inline review threads grouped by resolution status."""
    if not threads:
        print()
        print("=" * 60)
        print("INLINE REVIEW THREADS")
        print("=" * 60)
        print("  (none)")
        return

    unresolved = [t for t in threads if not t["isResolved"]]
    resolved = [t for t in threads if t["isResolved"]]

    print()
    print("=" * 60)
    print(
        "INLINE REVIEW THREADS "
        f"({len(unresolved)} unresolved, {len(resolved)} resolved)"
    )
    print("=" * 60)

    for label, group in [("UNRESOLVED", unresolved), ("RESOLVED", resolved)]:
        if not group:
            continue
        print()
        print(f"--- {label} THREADS ---")
        for i, thread in enumerate(group, 1):
            status_tags = []
            if thread["isResolved"]:
                status_tags.append("[RESOLVED]")
            else:
                status_tags.append("[UNRESOLVED]")
            if thread["isOutdated"]:
                status_tags.append("[OUTDATED]")

            path = thread.get("path", "unknown")
            line = thread.get("line") or thread.get("originalLine")
            line_info = f":{line}" if line else ""

            print()
            print(f"  Thread {i}: {' '.join(status_tags)} {path}{line_info}")

            comments = thread.get("comments", {}).get("nodes", [])
            for j, c in enumerate(comments):
                author = (c.get("author") or {}).get("login", "ghost")
                print(f"    [{author} at {c['createdAt']}]")

                diff_hunk = c.get("diffHunk", "")
                if diff_hunk and j == 0:
                    for hunk_line in diff_hunk.split("\n"):
                        print(f"      {hunk_line}")

                body = (c.get("body") or "").strip()
                if body:
                    for body_line in body.split("\n"):
                        print(f"    {body_line}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch PR review data via gh CLI and GraphQL"
    )
    parser.add_argument("pr_number", help="Pull request number (e.g. 1234)")
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository in owner/repo form (default: current repo)",
    )
    args = parser.parse_args()

    pr_data = fetch_pr_metadata(args.pr_number, args.repo)
    threads = fetch_review_threads(args.pr_number, args.repo)

    print_pr_header(pr_data)
    print_changed_files(pr_data.get("files", []))
    print_pr_body(pr_data.get("body"))
    print_top_level_comments(pr_data.get("comments", []))
    print_review_summaries(pr_data.get("reviews", []))
    print_review_threads(threads)


if __name__ == "__main__":
    main()
