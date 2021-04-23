#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import git
import itertools
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

Revision = Union[str, git.TagReference]


def get_last_release(
        repo: git.Repo,
        head_rev: Revision = 'HEAD') -> Optional[git.TagReference]:
    """Retrieve the release closest to the `head_rev` in its git history

    Returns:
      The tag representing the latest release, as measured by number of commits
      to the `head_rev`, or `None` if no release is in the history of
      `head_rev`.
    """
    def is_version_tag(tag):
        return str(tag).startswith('v')

    def distance_from_head(tag):
        return sum(1 for _ in repo.iter_commits(rev=f'{tag}..{head_rev}',
                                                first_parent=True))

    def is_in_history(tag):
        return repo.is_ancestor(ancestor_rev=tag, rev=head_rev)

    try:
        return min(filter(is_in_history, filter(is_version_tag, repo.tags)),
                   key=distance_from_head)
    except ValueError:
        return None


@dataclass
class PullRequest:
    id: int
    title: str
    author: str
    url: Optional[str] = None
    group: Optional[str] = None
    upgrade_instructions: Optional[str] = None


# These should correspond to labels on GitHub
PULL_REQUEST_GROUPS = ['major new feature', None, 'bugfix']


def get_merged_pull_requests(repo: git.Repo, from_rev: Revision,
                             to_rev: Revision) -> List[PullRequest]:
    """Parses list of merged PRs from merge commits.

    Parses merge commits in the repository between the revisions `from_rev` and
    `to_rev`. This is faster than querying GitHub and we can filter by commit
    SHAs instead of date.

    Returns:
      Merged pull-requests, ordered from least-recently merged to most-recently
      merged.
    """
    merge_commit_msg_pattern = '^Merge pull request #([0-9]+) from (.+)/'
    merged_prs = []
    for commit in repo.iter_commits(rev=f'{from_rev}..{to_rev}',
                                    first_parent=True):
        merge_commit_match = re.match(merge_commit_msg_pattern,
                                      commit.message,
                                      flags=re.MULTILINE)
        if not merge_commit_match:
            continue
        merged_prs.append(
            PullRequest(id=int(merge_commit_match.group(1)),
                        title=' '.join(commit.message.splitlines(False)[2:]),
                        author=merge_commit_match.group(2)))
    return merged_prs[::-1]


def get_upgrade_instructions(pr_description: str) -> Optional[str]:
    """Parse a section labeled "Upgrade instructions" from the PR description.

    This function looks for a section in the PR description that is enclosed in
    the HTML-comments `<!-- UPGRADE INSTRUCTIONS -->`. For example:

    ```md
    ### Upgrade instructions

    <!-- UPGRADE INSTRUCTIONS -->
    - Add the option `Evolution.InitialTime` to evolution input files. Set it
      to the value `0.` to keep the behavior the same as before.
    <!-- UPGRADE INSTRUCTIONS -->
    ```
    """
    FENCE_PATTERN = '<!-- UPGRADE INSTRUCTIONS -->'
    match = re.search(FENCE_PATTERN + '(.*)' + FENCE_PATTERN,
                      pr_description,
                      flags=re.DOTALL)
    if match is None:
        return None
    match = match.group(1).strip()
    if not match or match.isspace():
        return None
    return match


def compile_release_notes(merged_prs: List[PullRequest]) -> str:
    """Compile nicely-formatted release-notes.

    Args:
      merged_prs: The list of merged PRs since the last release, ordered from
        least-recently merged to most-recently merged. The format of the
        release notes depends on what information is available for the PRs.

    Returns:
      The Markdown-formatted release notes, ready to write to a file or print
      to screen.
    """
    # Sort PRs into their groups, ordered first by group then by the order they
    # were merged
    grouped_prs = itertools.groupby(
        sorted(merged_prs,
               key=lambda pr:
               (PULL_REQUEST_GROUPS.index(pr.group), merged_prs.index(pr))),
        key=lambda pr: pr.group)

    # Get set of distinct PR authors. Maintain the order by using a dict, which
    # is guaranteed to maintain the insertion order since Py 3.7
    pr_authors = list(dict.fromkeys([pr.author for pr in merged_prs]))

    def format_pr_link(pr):
        return f"#{pr.id}" if pr.url is None else f"[#{pr.id}]({pr.url})"

    def format_list_of_prs(prs):
        return [f"- {pr.title} ({format_pr_link(pr)})" for pr in prs]

    release_notes_content = []

    prs_with_upgrade_instructions = list(
        filter(lambda pr: pr.upgrade_instructions, merged_prs))
    if len(prs_with_upgrade_instructions) > 0:
        release_notes_content += ["## Upgrade instructions", ""]
        for pr in prs_with_upgrade_instructions:
            release_notes_content += [
                f"**From {format_pr_link(pr)} ({pr.title}):**", "",
                pr.upgrade_instructions, ""
            ]

    release_notes_content += [
        f"## Merged pull-requests ({len(merged_prs)})", ""
    ]
    if len(merged_prs) > 0:
        for group, prs_iterator in grouped_prs:
            group_header = {
                'major new feature': "Major new features",
                'bugfix': "Bugfixes",
                None: "General changes",
            }[group]
            prs = list(prs_iterator)
            release_notes_content += (
                [f"**{group_header} ({len(prs)}):**", ""] +
                format_list_of_prs(prs) + [""])
        release_notes_content += ([
            f"Contributors ({len(pr_authors)}): " +
            ", ".join([f"@{pr_author}" for pr_author in pr_authors])
        ] + [""])
    else:
        release_notes_content += ["_None_", ""]

    return '\n'.join(release_notes_content)


if __name__ == "__main__":
    # The release notes always refer to the repository that contains this file
    repo = git.Repo(__file__, search_parent_directories=True)

    import argparse
    parser = argparse.ArgumentParser(
        description=("Compile release notes based on merged pull-requests. "
                     f"Repository: {repo.working_dir}."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output',
        '-o',
        required=False,
        help=(
            "Name of the output file, e.g. 'release_notes.md'. If you omit "
            "this argument the Markdown-formatted output will be printed "
            "to stdout. Hint: Pipe the output to a CLI Markdown-renderer to "
            "improve readability, e.g. https://github.com/charmbracelet/glow."
        ))
    parser.add_argument(
        '--from',
        required=False,
        dest='from_rev',
        help=("Git revision that marks the last release, used as starting "
              "point for the release notes. Can be any commit SHA or tag. "
              "Defaults to the last release in the git history of "
              "the '--to' revision."))
    parser.add_argument(
        '--to',
        required=False,
        dest='to_rev',
        default='HEAD',
        help=("Git revision that marks this release. Can be any commit SHA "
              "or tag. Defaults to 'HEAD'."))
    parser.add_argument(
        '--github-repository',
        required=False,
        default='sxs-collaboration/spectre',
        help=("GitHub repository associated with pull-request IDs in merge "
              "commits."))
    parser_github_auth = parser.add_mutually_exclusive_group(required=False)
    parser_github_auth.add_argument(
        '--github-token',
        required=False,
        help=
        ("Access token for GitHub queries. Refer to the GitHub documentation "
         "for instructions on creating a personal access token."))
    parser_github_auth.add_argument(
        '--no-github',
        action='store_true',
        help=("Disable GitHub queries, working only with the local "
              "repository."))
    parser_logging = parser.add_mutually_exclusive_group(required=False)
    parser_logging.add_argument('-v',
                                '--verbose',
                                action='count',
                                default=0,
                                help="Verbosity (-v, -vv, ...)")
    parser_logging.add_argument('--silent',
                                action='store_true',
                                help="Disable any logging")
    args = parser.parse_args()

    # Set the log level
    logging.basicConfig(
        level=logging.CRITICAL if args.silent else (logging.WARNING -
                                                    args.verbose * 10))

    # Default `from_rev` to last release
    if args.from_rev is None:
        args.from_rev = get_last_release(repo=repo, head_rev=args.to_rev)
        logging.info(f"Last release is: {args.from_rev}")

    # Retrieve list of merged PRs since last release
    merged_prs = get_merged_pull_requests(repo=repo,
                                          from_rev=args.from_rev,
                                          to_rev=args.to_rev)
    logger.info("Merged PRs since last release:\n{}".format('\n'.join(
        map(str, merged_prs))))

    # Try to query GitHub for further information on the merged PRs
    if not args.no_github:
        import github
        gh = github.Github(args.github_token)
        gh_repo = gh.get_repo(args.github_repository)
        if args.silent:
            prs_iterator = iter(merged_prs)
        else:
            import tqdm
            prs_iterator = tqdm.tqdm(merged_prs,
                                     desc="Downloading PR data",
                                     unit="PR")
        for pr in prs_iterator:
            # First, download data
            pr_gh = gh_repo.get_pull(pr.id)
            # Update the title (it may have changed since the PR was merged)
            pr.title = pr_gh.title
            # Update the author (the GitHub username may have changed)
            pr.author = pr_gh.user.login
            # Add missing metadata to PR
            pr.url = pr_gh.html_url
            # Add group information to PR
            labels = [label.name for label in pr_gh.labels]
            for group in PULL_REQUEST_GROUPS:
                if group is None:
                    continue
                if group in labels:
                    pr.group = group
                    break
            # Add upgrade instructions to PR
            pr.upgrade_instructions = get_upgrade_instructions(pr_gh.body)

    # Assemble everything into a nicely formatted string
    release_notes_content = compile_release_notes(merged_prs=merged_prs)

    # Output
    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write(release_notes_content)
        logging.info(f"Release notes written to file: '{args.output}'")
    else:
        logging.info("Compiled release notes:")
        print(release_notes_content)
