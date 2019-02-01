\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Travis CI {#travis_guide}

\tableofcontents

# Testing SpECTRE with Travis CI {#travis_ci}

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

## What is tested {#what-is-tested}

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

## How to perform the checks locally {#perform-checks-locally}

Before pushing to GitHub and waiting for Travis to perform the checks it is
useful to perform at least the following tests locally:
- **Unit tests:** Perform a `make test-executables` and then execute `ctest` to
  run all unit tests. As for `make` you can append a `-jN` flag to `ctest` to
  run in parallel on `N` cores. To run only a subset of the tests you can use
  one of the other keywords that the tests are labeled with, such as `ctest -L
  datastructures`. To run only particular tests you can also execute `ctest -R
  TEST_NAME` instead, where `TEST_NAME` is a regular expression matching the
  test identifiers such as `Unit.DataStructures.Mesh`. Pass the flag
  `--output-on-failure` to get output from failed tests. Consult `ctest -h` for
  further options.
- **clang-tidy:** In a clang build directory, run `make clang-tidy
  FILE=SOURCE_FILE` where `SOURCE_FILE` is a relative or absolute path to a
  `.cpp` file. To perform this check for all source files that changed in your
  pull request, `make clang-tidy-hash HASH=UPSTREAM_HEAD` where `UPSTREAM_HEAD`
  is the hash of the commit that your pull request is based on, usually the
  `HEAD` of the `upstream/develop` branch.
- **IWYU:** Also just for the changed files in a pull request run `make
  iwyu-hash HASH=UPSTREAM_HEAD`. Since IWYU requires `USE_PCH=OFF` you can
  create a separate build directory and append `-D USE_PCH=OFF` to the usual
  `cmake` call. Note that it is very easy to incorrectly install IWYU (if not
  using the Docker container) and generate nonsense errors.
- **Documentation:** To render the documentation for the current state
  of the source tree the command `make doc` (or `make doc-check` to
  highlight warnings) can be used, placing its result in the `docs`
  directory in the build tree.  Once code has been made into a pull
  request to GitHub, the documentation can be rendered locally using
  the `tools/pr-docs` script.  To view the documentation, simply open the
  `index.html` file in the `html` subdirectory in a browser. Some functionality
  requires a web server (e.g. citation popovers), so just run a
  `python3 -m http.server` in the `html` directory to enable this.

## Travis setup {#travis-setup}

* The `gcc Debug` build runs code coverage for each Travis build.

## Troubleshooting {#travis-troubleshooting}

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

## Precompiled Headers and ccache {#precompiled-headers-ccache}

Getting ccache to work with precompiled headers on TravisCI is a little
challenging. The header to be precompiled is
`${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp` and is symbolically linked to
`${SPECTRE_BUILD_DIR}/SpectrePch.hpp`. The configuration that seems to work is
specifying the environment variables:

\code{.sh}
CCACHE_COMPILERCHECK=content
CCACHE_EXTRAFILES="${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp"
CCACHE_IGNOREHEADERS="${SPECTRE_BUILD_DIR}/SpectrePch.hpp:${SPECTRE_BUILD_DIR}/SpectrePch.hpp.gch"
\endcode

On macOS builds we haven't yet had success with using `ccache` with a
precompiled header. We disable the precompiled header and build in debug mode
only to have reasonable build times.

## Build Stages {#travis-build-stages}

In order to avoid timeouts we build SpECTRE in various stages, carrying over the
ccache from one stage to the next. This allows us to avoid recompiling the code
in the next stage (there is some small overhead from running ccache instead of
not doing anything at all). The first stage builds all the SpECTRE libraries but
none of the executables or testing libraries. The second stage builds builds the
test executables, runs the tests, and also runs ClangTidy, include-what-you-use,
and various other checks. Another stage could be added that builds some of the
test libraries if necessary.

## Caching Dependencies on macOS Builds {#caching-mac-os}

On macOS builds we cache all of our dependencies, like LIBXSMM and
Charm++. These are cached in `$HOME/mac_cache`. Ultimately this saves about
10-12 minutes even when compared to using ccache to cache the object files from
building the dependencies. We also cache `$HOME/Library/Caches/Homebrew`, which
is where Homebrew keeps the downloaded formulas. By caching the Homebrew bottles
we are able to avoid brew formulas building from source because a tarball of the
package was not available at the time.
