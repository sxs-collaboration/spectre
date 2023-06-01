\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Automatic versioning {#dev_guide_automatic_versioning}

\tableofcontents

Our automated tests can tag and publish a release automatically when requested.
The automation is implemented as a [GitHub
Actions](https://docs.github.com/actions) workflow in the file
`.github/workflows/Tests.yaml`.

## Creating releases

The GitHub workflow responsible for automated testing also allows creating a
release when dispatched manually. To create a release (roughly on the first
Monday of each month and/or when needed for publications), follow these
instructions:

1. **Check for bugs on `develop`.** If you are aware of any major bugs in the
   code on `develop`, check with the (other) core developers if it would be
   better to wait with the release until those bugs are fixed.
2. **Check the release notes.** Find the latest run of the [`Tests` workflow
   on the `develop` branch](https://github.com/sxs-collaboration/spectre/actions/workflows/Tests.yaml?query=branch%3Adevelop). Select the "Files and formatting" job and the
   "Print release notes" step within. Go through the release notes, checking
   the following in particular:
   - Label bugfix PRs with the `bugfix` label so they appear in the "Bugfixes"
     section and don't clutter the list.
   - Label PRs that you want to highlight with the `new feature` label, so they
     appear at the top. The selection is up to you.
   - Add upgrade instructions to PRs where they are missing, to help other
     developers with rebasing their code.
3. **Dispatch the release.** Go to the GitHub Actions page for the [`Tests`
   workflow](https://github.com/sxs-collaboration/spectre/actions/workflows/Tests.yaml?query=branch%3Adevelop).
   [Select "Run workflow"](https://docs.github.com/actions/managing-workflow-runs/manually-running-a-workflow)
   and the `develop` branch. Type in a valid release version name, such as
   "2021.05.04", and hit "Run workflow". The release will only be created if
   if the version name matches the format defined in
   \ref versioning_and_releases _and_ if it matches the current date. Therefore,
   if you wait too long with the next step, the release will fail and you have
   to dispatch it again.

   It is probably best to hold off merging PRs while the release workflow runs,
   to avoid possible conflicts. Therefore, warn the (other) core devs.
4. **Ask for approval.** Once the unit tests are complete, the release workflow
   will wait for approval before publishing the release on GitHub and Zenodo.
   Notify the (other) core devs so one of them can review and approve the
   release (don't approve the release yourself). You can point them to these
   guidelines:

   Guidelines for reviewing a release:
   - Check for bugs on `develop` (see point 1. above).
   - Check the release notes (see point 2. above).

## Release notes

The release notes are compiled automatically based on the activity in the
repository since the last release. They will contain a list of merged
pull-requests. Pull-requests labeled "new feature" or "bugfix" on GitHub
will be classified as such in the release notes.

The script `tools/CompileReleaseNotes.py` generates the release notes and can
also be invoked manually with a Python 3 interpreter to retrieve an overview of
what happened in the repository recently. The script requires you install
`GitPython`, `PyGithub` and `tqdm` in your Python environment.
