\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# CSCS Continuous Integration {#cscs_ci_guide}

\tableofcontents

# Overview {#cscs_ci_overview}

In addition to the tests that run on GitHub-hosted runners (see \ref
github_actions_guide), SpECTRE runs a short binary black hole simulation on HPC
hardware at the [Swiss National Supercomputing Centre
(CSCS)](https://docs.cscs.ch/services/cicd/). It runs on the CPU-only "Eiger"
cluster of the Alps infrastructure.

The pipeline runs on **every merge to `develop`, but not on pull requests**,
because it occupies a compute node and is billed against a CSCS project
allocation. It catches regressions that the GitHub Actions tests cannot: the
executables there only ever parse the binary black hole input files, they never
run a simulation.

The pipeline has four jobs:

0. `seed ccache image` creates the image tag that the build takes its ccache
   from, if it does not exist yet. It does nothing on every run after the
   first.
1. `build image` compiles `EvolveGhBinaryBlackHole`, `SolveXcts` and the
   `spectre` command-line interface into a container image, starting from the
   `sxscollaboration/spectre:dev` image that also serves the GitHub Actions
   tests.
2. `bbh regression` runs the simulation on one Eiger node and checks its
   output.
3. `update ccache tag` moves a tag onto the image that was just built, so the
   next run starts from a warm ccache.

# What the regression test does {#cscs_ci_test}

`.cscs-ci/RunBbhRegression.sh` runs two steps of the binary black hole pipeline
in `support/Pipelines/Bbh/` inside the Slurm allocation:

1. `spectre bbh generate-id` solves the XCTS equations for an equal-mass,
   non-spinning binary at low resolution and then finds the two apparent
   horizons.
2. `spectre bbh start-inspiral` evolves that initial data for a few `M`.

Both steps run with `--no-schedule`, which makes the CLI launch the executables
directly rather than submitting them with `sbatch`. This matters because the
CSCS runner has already allocated the compute node, and the Slurm tools are not
reachable from inside the container.

`.cscs-ci/CheckBbhRegression.py` then checks the output. The most important
check is that the evolution reached the requested final time: an evolution that
stops early, for instance because it ran out of wallclock time, still exits with
a zero exit code. The other checks cover the horizon masses and spins from the
initial data, the constraint violations during the evolution, and that the
apparent horizon finders ran. The thresholds are deliberately loose regression
guards, not statements about physical accuracy — the test runs at a resolution
far below any production simulation.

# Performance {#cscs_ci_performance}

The check script also records normalized timings to `performance.json`, kept as
a job artifact:

- for the initial data, the wall time and the number of Newton iterations, kept
  apart because a change in the iteration count is a solver regression while a
  change in the time per iteration is a performance regression;
- for the evolution, the start-up cost and the stepping cost separately, and
  `M/hr` (simulation time per wall hour), the figure of merit normally used for
  binary black hole runs.

They are normalized because the amount of work is not fixed: with local time
stepping and control systems the number of slabs needed to reach the final time
varies between runs, so raw wall time would conflate a slower code with a run
that did more work.

Nothing is asserted about these numbers yet. Setting a threshold requires
knowing the run-to-run spread first, which needs a handful of green runs on the
same hardware; picking one before that would only produce false failures. Be
realistic about what this can detect: on a single shared node, with a run this
short, only step changes are visible — an accidental quadratic algorithm, a
lost optimization flag, a debug assertion left enabled. It cannot see anything
below roughly 20-30%, and it cannot see parallel-scaling regressions at all,
because the test runs a few hundred elements on one node rather than the few
thousand across several nodes that a production run uses.

# Files {#cscs_ci_files}

Everything that lives in this repository is in the `.cscs-ci/` directory:

- `Eiger.yml` is the pipeline definition (GitLab CI syntax).
- `Dockerfile.eiger` is the recipe for the container image. The image is not
  published; it is pushed to the CSCS-internal registry and expires there. The
  images we publish are defined in `containers/Dockerfile.buildenv`.
- `Dockerfile.retag` is a two-line recipe used to move the ccache tag.
- `RunBbhRegression.sh` is the test.
- `CheckBbhRegression.py` checks its output and records the timings.

The rest of the configuration is **not** in this repository. The path to the
entry point, the list of CI-enabled branches, the list of trusted users, the
GitHub notification token, the webhook and the Slurm account all live on the
[CSCS CI setup page](https://cicd-ext-mw.cscs.ch/ci/overview).

# Reading the results {#cscs_ci_results}

The CSCS middleware reports the pipeline status back to GitHub as a commit
status, so a merge commit on `develop` shows a green or red mark that links to
the pipeline. Note this is a commit status and not a GitHub Actions check, so it
does not appear in the "Actions" tab.

The job log contains the full output of the executables. The pipeline also
keeps the generated input files and the reduction data (converted to text with
`spectre extract-dat`) as artifacts, including when the run failed. The volume
data is too large to keep.

# Running the test yourself {#cscs_ci_reproduce}

The test only needs the `spectre` CLI, `SolveXcts` and `EvolveGhBinaryBlackHole`
in the `PATH`, so you can run it in any build directory:

```sh
PATH=$SPECTRE_BUILD_DIR/bin:$PATH \
  SPECTRE_NUM_PROCS=4 SPECTRE_FINAL_TIME=1.0 \
  ./.cscs-ci/RunBbhRegression.sh /tmp/BbhRegression
```

To reproduce the CI environment more closely, build the container image and run
the test inside it. Your machine is probably not an AMD EPYC Rome, so override
the architecture: the base image is multi-arch and Docker builds natively for
your machine, so use `x86-64` on an Intel or AMD machine and `armv8-a` on Apple
Silicon.

```sh
docker build -f .cscs-ci/Dockerfile.eiger --build-arg NUM_PROCS=8 \
  --build-arg OVERRIDE_ARCH=armv8-a -t spectre:cscs-ci .
docker run --rm -it spectre:cscs-ci bash -c \
  'SPECTRE_NUM_PROCS=8 SPECTRE_FINAL_TIME=1.0 \
   /work/spectre/.cscs-ci/RunBbhRegression.sh /tmp/BbhRegression'
```

The `SPECTRE_NUM_PROCS`, `SPECTRE_FINAL_TIME`, `SPECTRE_ID_L`, `SPECTRE_ID_P`,
`SPECTRE_EV_L` and `SPECTRE_EV_P` environment variables tune the resolution and
the length of the run. `.cscs-ci/Eiger.yml` sets the first two; the others use
the defaults documented in `.cscs-ci/RunBbhRegression.sh`.

If you are on the trusted-users list you can also trigger the pipeline from a
pull request by writing a comment with:

```
cscs-ci run eiger;CSCS_CI_FORCE_RUN=1
```

# Things to be aware of {#cscs_ci_pitfalls}

- **The published `dev` image can lag behind `develop`.** It is rebuilt by hand
  with the `BuildDockerContainer` workflow, so after a change to the minimum
  compiler, CMake or Python version the image may no longer be able to build
  `develop` until someone reruns that workflow. This is why the CI image
  compiles with GCC rather than Clang. The image also currently ships
  Charm++ 7.0.0, which means dynamic h-refinement is unavailable: today that is
  harmless because the binary black hole input files declare no AMR criteria,
  but a run that triggers h-coarsening on more than one core would abort (see
  `src/ParallelAlgorithms/Amr/Actions/CreateParent.hpp`).
- **Do not set `MACHINE` when configuring the CI image.** That would make
  `spectre.support.Schedule` default to submitting jobs with `sbatch`, which is
  not reachable from inside the container.
- **The architecture must be pinned with `OVERRIDE_ARCH`.** The default is
  `-march=native`, and the Kubernetes pod that builds the image is a different
  machine from the Eiger compute node.
- **Charm++ in the container is a `multicore` build**, i.e. a single process
  with worker threads that cannot span nodes. So the test requests one Slurm
  task on one node and sizes the run with `+p`.
- **Registry layer caching does not help.** Copying the sources into the image
  invalidates the compile layer on every commit. Instead the ccache directory is
  carried forward inside the image itself, see `.cscs-ci/Dockerfile.eiger`.

# Setting up the CI service {#cscs_ci_setup}

These steps cannot be done in this repository and need someone with both CSCS
and GitHub organization access. They are recorded here in case the pipeline ever
has to be set up again.

1. Open a CSCS Service Desk ticket to register the repository, naming the target
   system (Eiger / zen2), the expected usage and the project that Slurm
   accounting should go to.
2. Create a GitHub token with permission to write commit statuses and enter it
   as the notification token on the CI setup page. The page refuses to save
   while the token is invalid. Use a *classic* token; the CSCS documentation
   discourages fine-grained tokens.
3. Add the webhook shown on the setup page to the GitHub repository. Enable the
   push, pull request and issue comment events — GitHub's default of "push only"
   silently disables the `cscs-ci run` comment trigger.
4. Enter the FirecREST credentials and the Slurm account.
5. Create a pipeline with entry point `.cscs-ci/Eiger.yml`, set its CI-enabled
   branches to `develop`, and add the core developers to its trusted users.
   Pull requests from forks only trigger a pipeline for trusted users, so
   ordinary pull requests never consume the allocation.

The CI-enabled branches are the gate that decides whether a pipeline runs at
all; the `workflow: rules:` in `.cscs-ci/Eiger.yml` can only narrow that
further, never widen it. So to try the pipeline out on a branch you have to add
that branch in *both* places: to the CI-enabled branches on the setup page, and
to the branch names matched by the `workflow: rules:`.
