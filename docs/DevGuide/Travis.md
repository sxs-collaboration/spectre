\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Travis CI {#travis_guide}

SpECTRE uses
[TravisCI](https://travis-ci.org/sxs-collaboration/spectre) for
testing the code.  Multiple build jobs (described below) are launched
each time a pull request is submitted or updated.  In addition, a
daily cron job is run.  Travis will also launch these build jobs each
time you push to a branch on your fork of SpECTRE if you enable
it. (Go to Travis, click on your name in the upper right corner, and
then towards the bottom, enable testing on your fork of SpECTRE.)

For pull requests, you can view the Travis CI build by clicking on
`Details` next to the `Travis build failed` or `Travis build
succeeded` in the section for `All checks have failed` or `All checks
have passed` towards the bottom of the pull request.  You can login to
Travis using your GitHub credentials in order to cancel your builds,
and you should do so if you update your pull request while it is
building.

## What is tested

The Travis report lists the build jobs which will each have either a
green check mark if it passes, a red `X` if it has failed, or a yellow
binary if the build is in progress.  Clicking on a build job will
display the log of the build.

The following build jobs are launched:
* CHECK_COMMITS runs the script `tools/CheckCommits.sh` and fails the build if
any casing of the words in the list below is the first word of the commit
message.  This allows developers to flag their commits with these keywords to
indicate that a pull request should not be merged in its current state.
  - fixup
  - wip (for work in progress)
  - fixme
  - deleteme
  - rebaseme
  - testing
  - rebase
* CHECK_FILES runs the script `tools/CheckFiles.sh` (which also runs the script
`tools/FileTestDefs.sh`) and fails the build if any of the following checks are
true:
  - A `c++` file (i.e. `*.hpp`, `*.cpp`, or `*.tpp` file) contains a
  line over 80 characters
  - A file contains a tab character
  - A file has white space at the end of the line
  - A file has a carriage return character
  - A file is missing the license line
  - A `*.hpp` file is missing `pragma once`
  - A file does not end with a newline
  - A `c++` file includes any of the following headers
    * `<iostream>`  (useless when running in parallel)
    * `Utilities/TmplDebugging.hpp`  (used only for debugging)
  - A `c++` file contains a `namespace` ending in `_details` (use `_detail`)
  - A `c++` file contains a `struct TD` or `class TD` (used only for debugging)
  - A `c++` file contains `std::enable_if` (use `Requires` instead)
  - A `c++` file contains additional text after `/*!` (does not render correctly
  in doxygen)
  - A `c++` file contains `Ls` (use `List` instead)
  - A `c++` test uses either:
    * `TEST_CASE` (use `SPECTRE_TEST_CASE` instead)
    * `Approx` (use `approx` instead)
* RUN_CLANG_TIDY runs the script `.travis/RunClangTidy.sh` which runs
`clang-tidy` on all files which were modified.  This is done for both `Release`
and `Debug` builds.
* TEST_CHECK_FILES runs `tools/CheckFiles.sh --test` which tests the checks
performed in the CHECK_FILES build.
* The other builds compile the code and run the tests for both
`Release` and `Debug` builds, for the `gcc` and `clang` compilers
using a Linux OS, and the `AppleClang` compiler for `OS X`.
* The `gcc Debug` build will fail if there are `doxygen` warnings.

## Travis setup

* The `gcc Debug` build runs code coverage for each Travis build.

## Troubleshooting

* Occasionally, a build job will fail because of a problem with Travis
(e.g. it times out).  Clicking on the circular arrow icon on the far
right of the build job will restart it.  All build jobs can be
restarted by clicking on the `Restart build` button to the right of
the pull request description.
* Travis caches some things between builds.  Occasionally this may
cause a problem leading to strange build failures.  If you suspect
this may be the case, get a SpECTRE owner to click on the `More
options` button in the top right, choose `Caches` and delete the cache
for your pull request.
