<!--
Distributed under the MIT License.
See LICENSE.txt for details.
-->

# CSCS continuous integration

Each `.yml` file in this directory is the entry point of a pipeline that runs
on the Alps infrastructure of the
[Swiss National Supercomputing Centre (CSCS)](https://docs.cscs.ch/services/cicd/).
The pipelines run on every merge to `develop`, but not on pull requests,
because they are billed against a CSCS project allocation. A trusted user can
run a pipeline on a pull request with a comment like
`cscs-ci run regression-tests;CSCS_CI_FORCE_RUN=1`. The result is reported
back to GitHub as a commit status, not as a GitHub Actions check. When a
pipeline starts failing on `develop`, `.github/workflows/CscsCiFailures.yaml`
comments on the pull request that was merged.

The container images build on the published `sxscollaboration/spectre` images,
which are rebuilt by hand with the `BuildDockerContainer` workflow. After a
change to the minimum version of a dependency, rerun that workflow or the
pipelines here may fail to build.

Things to know when writing a pipeline:

- The images are pushed under `$CSCS_REGISTRY_PATH/public/`, so they can be
  pulled from a CSCS login node to debug a failed run.
- A job's `timeout` includes the time waiting in the Slurm queue. The allocation
  itself is limited by `SLURM_TIMELIMIT`, which must be a quoted `"HH:MM:SS"`:
  unquoted, the runner converts it to seconds, but Slurm reads a bare integer as
  minutes.

## Setting up the CI service

Everything except the pipeline definitions is configured on the
[CSCS CI setup page](https://cicd-ext-mw.cscs.ch/ci/overview), not in this
repository. Follow the
[CSCS instructions](https://docs.cscs.ch/services/cicd/#enable-ci-for-your-project)
to register the repository and set it up. Then create one pipeline per `.yml`
file in this directory, named after the file in kebab-case (e.g.
`regression-tests` for `RegressionTests.yml`). Set its CI-enabled branches to
`develop` and add the core developers to its trusted users.

The CI-enabled branches decide whether a pipeline runs at all. The
`workflow: rules:` in each `.yml` file can only narrow that further: they are a
second line of defense for pull requests opened from a branch of the main
repository. So to try a pipeline out on a branch, add it in both places. Note
that `CI_PIPELINE_SOURCE` is always "trigger" on this service and that
`rules: changes` does not work, but matching on the branch name does.
