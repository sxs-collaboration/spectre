#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import textwrap
import unittest

import git
import yaml

from tools.CompileReleaseNotes import (
    PullRequest,
    compile_release_notes,
    get_last_release,
    get_merged_pull_requests,
    get_upgrade_instructions,
)


class TestCompileReleaseNotes(unittest.TestCase):
    def test_get_last_release(self):
        repo = git.Repo(path=__file__, search_parent_directories=True)
        with open(
            os.path.join(repo.working_dir, "Metadata.yaml"), "r"
        ) as open_metadata_file:
            metadata = yaml.safe_load(open_metadata_file)
        expected_tag_name = "v" + metadata["Version"]
        last_release = get_last_release(repo, head_rev="HEAD")
        self.assertEqual(
            str(last_release),
            expected_tag_name,
            (
                f"The last release '{last_release}' doesn't match the expected "
                f"tag name '{expected_tag_name}' listed in the repo metadata."
            ),
        )
        also_last_release = get_last_release(repo, head_rev=last_release)
        self.assertEqual(
            last_release,
            also_last_release,
            (
                f"Should find the last release '{last_release}' when HEAD is"
                f" the release tag, but found '{also_last_release}'."
            ),
        )
        earlier_release = get_last_release(
            repo, head_rev=str(last_release) + "~1"
        )
        self.assertNotEqual(
            earlier_release,
            last_release,
            (
                "Should find an earlier release when HEAD is before "
                f"'{last_release}', not the same release."
            ),
        )

    def test_get_merged_pull_requests(self):
        repo = git.Repo(path=__file__, search_parent_directories=True)
        last_release = get_last_release(repo)
        last_merge_commit = next(
            repo.iter_commits(rev=last_release, min_parents=2)
        )
        merged_prs = get_merged_pull_requests(
            repo,
            from_rev=(str(last_merge_commit) + "~1"),
            to_rev=last_merge_commit,
        )
        self.assertTrue(
            len(merged_prs) > 0,
            (
                "Failed to parse pull request corresponding to last merge "
                f"commit before release {last_release}: '{last_merge_commit}' "
            ),
        )

    def test_get_upgrade_instructions(self):
        upgrade_instructions = get_upgrade_instructions(textwrap.dedent("""\
            ### Upgrade instructions
            Here's what you should do when rebasing on this PR:
            <!-- UPGRADE INSTRUCTIONS -->
            - Add the option `Evolution.InitialTime` to evolution input files.
              Set it to the value `0.` to keep the behavior the same as before.
            <!-- UPGRADE INSTRUCTIONS -->
            """))
        self.assertEqual(
            upgrade_instructions,
            (
                "- Add the option `Evolution.InitialTime` to evolution input"
                " files.\n  Set it to the value `0.` to keep the behavior the"
                " same as before."
            ),
        )
        self.assertIsNone(
            get_upgrade_instructions(
                "<!-- UPGRADE INSTRUCTIONS --> <!-- UPGRADE INSTRUCTIONS -->"
            ),
            "Not all whitespace is stripped",
        )
        self.assertIsNone(
            get_upgrade_instructions(
                "<!-- UPGRADE INSTRUCTIONS -->\n<!-- UPGRADE INSTRUCTIONS -->"
            ),
            "Not all whitespace is stripped",
        )
        self.assertEqual(
            get_upgrade_instructions(
                "<!-- UPGRADE INSTRUCTIONS -->\n\n\n   Do this and that."
                "  \n\n<!-- UPGRADE INSTRUCTIONS -->"
            ),
            "Do this and that.",
            "Not all whitespace is stripped",
        )
        self.assertIsNone(get_upgrade_instructions(""))
        self.assertIsNone(get_upgrade_instructions(None))

    def test_compile_release_notes(self):
        self.assertEqual(
            compile_release_notes([]),
            textwrap.dedent("""\
                ## Merged pull-requests (0)

                _None_
                """),
        )
        pr1 = PullRequest(id=1, title="Add this", author="author1")
        pr2 = PullRequest(
            id=2,
            title="Also add this new feature",
            author="author2",
            url="https://github.com/2",
        )
        self.assertEqual(
            compile_release_notes([pr1, pr2]),
            textwrap.dedent("""\
                ## Merged pull-requests (2)

                **General changes (2):**

                - Add this (#1)
                - Also add this new feature ([#2](https://github.com/2))

                Contributors (2): @author1, @author2
                """),
        )
        major_pr1 = PullRequest(
            id=3,
            title="This is big",
            author="author2",
            group="new feature",
            upgrade_instructions="- Do this.\n- And that.",
        )
        major_pr2 = PullRequest(
            id=4, title="Another feature", author="author3", group="new feature"
        )
        bugfix_pr1 = PullRequest(
            id=5,
            title="Fixed this bug",
            author="author3",
            url="https://github.com/5",
            group="bugfix",
            upgrade_instructions="You'll have to rerun your simulation.",
        )
        bugfix_pr2 = PullRequest(
            id=6, title="Fixed another bug", author="author1", group="bugfix"
        )
        self.assertEqual(
            compile_release_notes(
                [pr1, major_pr1, pr2, bugfix_pr1, bugfix_pr2, major_pr2]
            ),
            textwrap.dedent("""\
                ## Upgrade instructions

                **From #3 (This is big):**

                - Do this.
                - And that.

                **From [#5](https://github.com/5) (Fixed this bug):**

                You'll have to rerun your simulation.

                ## Merged pull-requests (6)

                **New features (2):**

                - This is big (#3)
                - Another feature (#4)

                **General changes (2):**

                - Add this (#1)
                - Also add this new feature ([#2](https://github.com/2))

                **Bugfixes (2):**

                - Fixed this bug ([#5](https://github.com/5))
                - Fixed another bug (#6)

                Contributors (3): @author1, @author2, @author3
                """),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
