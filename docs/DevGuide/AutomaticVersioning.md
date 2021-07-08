\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Automatic versioning {#dev_guide_automatic_versioning}

Our automated tests can tag and publish a release automatically when requested.
The automation is implemented as a [GitHub
Actions](https://docs.github.com/actions) workflow in the file
`.github/workflows/Tests.yaml`.

## Creating releases

The GitHub workflow responsible for automated testing also allows creating a
release when dispatched manually. To create a release, follow the instructions
to manually run a workflow:

- [Manually running a workflow on GitHub](https://docs.github.com/actions/managing-workflow-runs/manually-running-a-workflow)

To create a release you will have to run the workflow "Tests" on the `develop`
branch _and_ type in a valid release version name. The workflow will only create
the release if the version name matches the format defined above and if it
matches the date at the time the "Release version" job runs.

## Release notes

The release notes are compiled automatically based on the activity in the
repository since the last release. They will contain a list of merged
pull-requests. Pull-requests labeled "new feature" or "bugfix" on GitHub
will be classified as such in the release notes.

The script `tools/CompileReleaseNotes.py` generates the release notes and can
also be invoked manually with a Python 3 interpreter to retrieve an overview of
what happened in the repository recently. The script requires you install
`GitPython`, `PyGithub` and `tqdm` in your Python environment.
