# Distributed under the MIT License.
# See LICENSE.txt for details.

from pathlib import Path
from typing import Optional


def check_spec_import(contains_commit: Optional[str] = None):
    """Check that the SpEC Python modules can be imported.

    Arguments:
      contains_commit: If specified, check that the SpEC repository contains
        the specified commit. This can be used as a version check to ensure
        that the installed SpEC version is compatible with the code that is
        importing it.
    """
    try:
        import Utils as spec_utils
    except ImportError:
        raise ImportError(
            "Importing from SpEC failed. Make sure you have pointed "
            "'-D SPEC_ROOT' to a SpEC installation when configuring the build "
            "with CMake."
        )
    if contains_commit:
        import subprocess

        spec_repo_dir = Path(spec_utils.__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", contains_commit, "HEAD"],
            cwd=spec_repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ImportError(
                "SpEC version mismatch: requested revision"
                f" {contains_commit} is not in the SpEC repository at"
                f" {spec_repo_dir}."
            )
