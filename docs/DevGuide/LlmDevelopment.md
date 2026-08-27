\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Coding agents and LLMs {#coding_agents_llms}

\tableofcontents

# Overview {#coding_agents_llms_intro}

SpECTRE ships infrastructure for running coding agents (Claude Code, OpenAI
Codex, GitHub Copilot) against the repository: a container image
(`containers/CodingAgents.def`), custom skills (e.g., `spectre-review`,
`spectre-fix-pr`, `spectre-fix-ci`, `spectre-fix-issue`), coding-standard rules
in `.claude/rules/`, and lifecycle hooks in `.claude/hooks/`. This page explains
how to build and run the container, configure the GitHub CLI, write a good
`CLAUDE.md`/`AGENTS.md`, use the skills, understand and extend the rules and
hooks, and adopt an effective agentic workflow.

This page assumes general familiarity with coding agents. As a quick refresher,
agents runs in one of a few modes:

- **Plan mode** — the agent researches and writes a plan but makes no edits.
- **Normal ("no" / ask) mode** — the agent proposes each action and asks
  before editing; useful for discussion and exploration.
- **Accept-edits mode** — edits are applied without per-edit confirmation.
- **Auto / bypass mode** — the agent runs tools without prompting.

Skills are invoked as slash (Claude Code) or dollar (OpenAI Codex) commands,
e.g. `/spectre-review` or `$spectre-review`. You can see a list
of spectre-related skills by typing `/spectre-` on an empty prompt.

There are plenty of great intro resources to using coding agents, e.g.
https://code.visualstudio.com/docs/agents/best-practices

# Building and running the container {#coding_agents_llms_container}

The container definition `containers/CodingAgents.def` bootstraps from the
`sxs-collaboration/spectre:dev` image and adds everything an agent needs:

- the Clang 21 toolchain plus `clangd-21` for C++ language-server support, GCC
  12 / `gfortran-12`, and `clang-tidy`;
- CUDA 12.6 (`nvcc`); Kokkos itself is fetched automatically by SpECTRE's CMake
  (`SPECTRE_KOKKOS=ON`, `SPECTRE_FETCH_MISSING_DEPS=ON`), not pre-installed;
- SpEC dependencies: PETSc, serial HDF5, FFTW, ParaView, `bear`, `git-annex`;
- Node.js 20 and the agent CLIs `@anthropic-ai/claude-code`, `@openai/codex`,
  and `@github/copilot`;
- a `/usr/local/bin/claude` wrapper that auto-installs the `clangd-lsp` and
  `pyright-lsp` plugins and sets `ENABLE_LSP_TOOL=1`, so the agent has C++ and
  Python code intelligence out of the box;
- you can also compile and run SpEC in the container.

## Building the image {#coding_agents_llms_container_build}

Build the image with Apptainer (the help section of the definition file
documents this):

```bash
apptainer build --fakeroot SpectreAi.sif ./containers/CodingAgents.def
```

To pin a specific Claude Code version, pass it through the build environment:

```bash
CLAUDE_CODE_VERSION=1.2.3 \
  apptainer build SpectreAi.sif containers/CodingAgents.def
```

## Running on Linux (Apptainer) {#coding_agents_llms_container_linux}

It is convenient to define a shell function that launches the container with
all the right binds. For example, the following shell functions launch the
container and bind various directories. You can place it in
your `~/.bashrc` (or `~/.zshrc`):

```bash
function claude-anthropic() {
    if [[ -z $1 ]]; then
        echo "Usage: claude-anthropic <number> [other_args]" >&2
        echo "The <number> binds the container's \$HOME/spectre to the" >&2
        echo "host's \$HOME/spectre_claude<number>, enabling multiple" >&2
        echo "simultaneous nonoverlapping sessions." >&2
        return 1
    fi
    local num=$1
    shift
    CONTAINER_TMP=$(mktemp -d /tmp/$USER-claude-anthropic-XXXXXX)
    mkdir -p "$CONTAINER_TMP"
    apptainer shell --contain \
              -B "$CONTAINER_TMP":/tmp \
              -B ~/spectre_claude${num}:$HOME/spectre \
              -B ~/SurrogateModeling:$HOME/SurrogateModeling \
              -B ~/Claudes/Anthropic/.claude:$HOME/.claude \
              -B ~/Claudes/Anthropic/.claude.json:$HOME/.claude.json \
              -B ~/.config/gh:$HOME/.config/gh:ro \
              -B ~/SpECTRE_AGENTS.md:$HOME/spectre/CLAUDE.md:ro \
              --add-caps CAP_NET_RAW ~/SpectreAi.sif "$@"
    rm -rf "$CONTAINER_TMP"
}
```

You then launch an agent on, say, your second checkout with
`claude-anthropic 2`. The flags do the following:

- `--contain` — isolate the container from the host filesystem and
  environment; only the directories you explicitly bind are visible inside.
  This prevents an agent from touching anything you did not deliberately
  expose.
- `-B "$CONTAINER_TMP":/tmp` — bind a freshly `mktemp`'d, per-session scratch
  directory as `/tmp`. The container sets `TMPDIR=/tmp`, so this gives each
  agent its own writable temp space, which is removed when the function
  returns.
- `-B ~/spectre_claude${num}:$HOME/spectre` — mount one of several numbered
  SpECTRE checkouts as the repo. Keeping separate worktrees
  (`~/spectre_claude1`, `~/spectre_claude2`, ...) lets several agents work in
  parallel without clobbering each other's build trees or branches.
- `-B ~/SurrogateModeling:$HOME/SurrogateModeling` — mount an additional
  project the agent needs (replace or drop to taste).
- `-B ~/Claudes/Anthropic/.claude:$HOME/.claude` and
  `-B ~/Claudes/Anthropic/.claude.json:$HOME/.claude.json` — persist the
  agent's configuration, credentials, and history across runs. Without these,
  `--contain` would hide them and you would re-authenticate every session.
- `-B ~/.config/gh:$HOME/.config/gh:ro` — share your host GitHub CLI
  authentication read-only (see \ref coding_agents_llms_gh).
- `-B ~/SpECTRE_AGENTS.md:$HOME/spectre/CLAUDE.md:ro` — overlay your personal
  agent instructions as `CLAUDE.md` inside the repo without committing them.
  The `:ro` (read-only) suffix means the agent cannot edit its own
  instructions.
- `--add-caps CAP_NET_RAW` — grant the raw-socket capability so tools like
  `ping` work for network diagnostics inside the container.
- `"$@"` — forward any extra arguments (e.g. an initial prompt or flags) to
  `apptainer shell`.

## Running on macOS (Apple Container) {#coding_agents_llms_container_macos}

On macOS with Apple's `container` tool, the following shell function is
convenient for launching the container. You may add it to your `~/.zshrc`:

```bash
function claude-container() {
    container run --rm -ti --memory 8g \
              -v ~/.claude:/home/claude/.claude \
              -v ~/.claude.json:/home/claude/.claude.json \
              -e GIT_NAME="$(git config --global user.name)" \
              -e GIT_EMAIL="$(git config --global user.email)" \
              -v ~/Research/spectre:/home/claude/work/spectre \
              -v ~/.config/gh:/home/claude/.config/gh:ro \
              -v ~/Research/gwsurrogate:/home/claude/work/gwsurrogate \
              -v ~/Research/gwsurrogate_experiment:/home/claude/work/gwsurrogate_experiment \
              -v ~/Research/surrogate_modeling:/home/claude/work/surrogate_modeling $@
}
```

The flags mirror the Linux setup:

- `container run --rm -ti` — run an ephemeral, interactive container that is
  removed (`--rm`) when you exit.
- `--memory 8g` — cap the container's memory usage.
- `-v host:container` — bind mounts, the Apple-Container equivalent of
  Apptainer's `-B`: persist `.claude`/`.claude.json` config, mount the
  `spectre` repo read-write, mount the `gh` config read-only, and mount any
  extra research projects.
- `-e GIT_NAME=... -e GIT_EMAIL=...` — pass your host git identity into the
  container as environment variables. Apple Container does not inherit your
  host git config, so this ensures commits are attributed correctly.
- `$@` — forward any extra arguments.

The cross-platform pattern is the same on both systems: keep the agent
configuration persistent, mount the repo read-write, mount the `gh` config
read-only, and give each session an isolated `/tmp`.

\note You could add `--ssh claude-spectre` to forward the SSH key/agent named
  `claude-spectre` so the agent can `git push` and perform signed git
  operations. However, since these operations may be destructive, it is
  discouraged.

# Configuring the GitHub CLI {#coding_agents_llms_gh}

The skills shell out to the GitHub CLI (`gh`) to fetch PRs, issues, and CI
logs, so `gh` must be authenticated. The `:ro` bind in the launch functions
shares your host authentication read-only — but the *first-time* setup must
happen somewhere `gh` can write its config. You have two options:

1. **Authenticate on the host once.** Run `gh auth login` on the host; the
   `:ro` bind then reuses that authentication inside the container. This is the
   simplest approach.
2. **Authenticate inside the container.** Temporarily drop the `:ro` so
   `~/.config/gh` is writable — change the bind to
   `-B ~/.config/gh:$HOME/.config/gh` (Linux) or
   `-v ~/.config/gh:/home/claude/.config/gh` (macOS) — run `gh auth login`,
   then switch back to `:ro` for normal use.

## Authenticating with a token {#coding_agents_llms_gh_token}

For agent/headless use, authenticating with a Personal Access Token is
recommended. Create a **classic** token at GitHub → Settings → Developer
settings → Personal access tokens → **Tokens (classic)**, then authenticate:

```bash
gh auth login --with-token < my_token.txt
# or non-interactively:
GH_TOKEN=ghp_xxx gh ...
```

Grant the **minimum** scopes:

- `repo` — read/write access to pull requests, issues, comments, and CI logs
  on both private and public repositories. This is the core scope the skills
  need.
- `read:org` — read organization membership (needed for `sxs-collaboration`).
- `workflow` — **only** add this if the agent will edit files under
  `.github/workflows/`.

Do **not** grant broader scopes such as `admin:*`, `delete_repo`,
`write:packages`, or user/gist scopes. Follow least privilege: an agent's token
should not be able to perform destructive organization or account operations.
Set an expiration, rotate the token periodically, and never commit it to the
repository.

# Example CLAUDE.md / AGENTS.md {#coding_agents_llms_claude_md}

The repository does **not** ship a `CLAUDE.md`/`AGENTS.md`; each user supplies
their own. One possible pattern is to keep your instructions in a file such as
`~/SpECTRE_AGENTS.md` and overlay it into the repo at run time via the
`-B ~/SpECTRE_AGENTS.md:$HOME/spectre/CLAUDE.md:ro` bind, so it is never
committed. `AGENTS.md` is the cross-tool equivalent read by Codex, Copilot, and
others; it can hold the same content (or be a symlink/copy of your
`CLAUDE.md`).

The following is a complete starting point you can copy and adapt:

```markdown
# Repository Guidelines

## Project Structure
Build in `./build` (never in the repo root). The main branch is `develop`,
not `main`. Directories matching `./build*` are not part of the repo and
should be ignored.

## Build, Test, and Development Commands
Configure once per build tree:
    cmake -S . -B ./build -GNinja \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CHARM_ROOT=${CHARM_ROOT} \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_CXX_COMPILER=clang++-21 \
      -D CMAKE_C_COMPILER=clang-21 \
      -D CMAKE_Fortran_COMPILER=gfortran-12
Build everything: `cmake --build ./build -j 12`
Build one target: `cmake --build ./build -j 12 --target Test_Foo`
Run tests: `ctest --test-dir ./build --output-on-failure -j 12 -R TEST_NAME`

## Coding Style
Coding standards live in `.claude/rules/`. After changes, verify hooks pass
with `git commit --dry-run`.

## Testing Guidelines
After the specific test passes, build `unit-tests` and run
`ctest --test-dir ./build -L unit --output-on-failure -j 12` to check for
regressions.

## Commit & PR Guidelines
Use short, imperative subject lines (e.g. "Add TensorYlmSphereToCart") and
reference issue numbers when available. PRs summarize motivation,
user-visible effects, and testing performed.
```

What makes a *good* `CLAUDE.md`/`AGENTS.md`:

- Short, skimmable, and imperative — the agent reads it every session, so keep
  the signal-to-noise ratio high.
- Exact, copy-pasteable configure/build/test commands with the real flags your
  project uses.
- Hard constraints stated as MUST/NEVER (e.g. "the main branch is `develop`",
  "build in `./build`, never the repo root").
- Pointers to detailed conventions (`.claude/rules/`) rather than inlining
  every rule.
- Kept current: prune stale instructions, and reference issues/PRs by number.
- Avoid duplicating what tooling already enforces (e.g. formatting is handled
  by a hook — see below).

# Skills {#coding_agents_llms_skills}

Skills are invoked as slash commands. The four SpECTRE skills are pre-approved
in `.claude/settings.json`, so they run without a permission prompt.

- **`/spectre-review [PR#] [clang-tidy] [coverage]`** — an orchestrated code
  review of a pull request or of local commits. It runs formatting checks, CI
  pre-checks, parallel style and bug-finding agents, optional `clang-tidy` and
  coverage passes, and produces a pruned final report. With **no** PR number it
  reviews the local commits on your current branch (the "local" option), and
  you can scope it to a specific commit in natural language, e.g.
  `/spectre-review local only review commit abc123`, `/spectre-review 1234`,
  `/spectre-review 1234 clang-tidy coverage`.
- **`/spectre-fix-pr PR_NUMBER [--repo OWNER/REPO]`** — fetch a PR's metadata
  and inline review threads (unresolved threads first) so the agent can work
  through reviewer feedback. Example: `/spectre-fix-pr 1234`.
- **`/spectre-fix-ci JOB_ID [--owner OWNER]`** — fetch a failed GitHub Actions
  job log with context windows around each failure. `JOB_ID` is the last
  number in the Actions URL (in `runs/24319442407/job/71002790711` it is
  `71002790711`). Example: `/spectre-fix-ci 71002790711`.
- **`/spectre-fix-issue ISSUE_NUMBER [--repo OWNER/REPO] [--full]`** — fetch a
  GitHub issue. It automatically detects "random failure" issues and emits the
  failing seeds, pointing the agent at the `RANDOM_FAILURE.md` investigation
  workflow (which prioritizes fixing the underlying numerical bug over
  loosening tolerances). Example: `/spectre-fix-issue 1727`.

# Rules and hooks {#coding_agents_llms_rules_hooks}

## Rules {#coding_agents_llms_rules}

Files in `.claude/rules/` are coding-standard documents auto-applied to
matching files by glob:

- `Cxx.md` — applies to `**/*.{hpp,cpp,tpp}`. Covers banned patterns (e.g.
  `#include <iostream>`, `std::enable_if`, logical operators), prefer-library
  guidance (use the existing `EagerMath` utilities instead of hand-rolled
  tensor loops), naming and style conventions, and test requirements.
- `CMake.md` — applies to `**/CMakeLists.txt`. Requires alphabetically ordered
  entries in `spectre_target_sources()`/`spectre_target_headers()` and use of
  the `${LIBRARY}` variable instead of hardcoded library names.

Add a new rule file when a language or area of the code has conventions an
agent repeatedly gets wrong, or when a review nit keeps recurring and is worth
encoding once rather than correcting every time.

## Hooks {#coding_agents_llms_hooks}

Scripts in `.claude/hooks/` run at points in the agent lifecycle. One hook is
enabled by default in `.claude/settings.json`:

- **`PostFormat.sh`** (`PostToolUse`, after `Edit`/`Write`) — auto-formats the
  modified file: `git-clang-format`/`clang-format` for C++, `black` and `isort`
  for Python.

Three more hooks live in the repo but are **off by default**; enable them in
your own `.claude/settings.local.json` (see below):

- **`SnapshotState.sh`** (`PreToolUse`) — records a baseline of the repository
  state on the first code-modifying tool call of a session.
- **`WarnCapture.sh`** (`PostToolUse`, after `Bash`) — scans command output
  for compiler warnings and blocks, forcing the agent to fix them before
  continuing.
- **`StopVerify.sh`** (`Stop`) — when the agent tries to finish, compares the
  current state against the baseline and, if there are changes, prompts the
  agent to run the unit and non-unit test suites before stopping. It uses the
  shared helper `Common.sh`.

Add a new hook when you want to enforce a project invariant *mechanically* —
running a formatter, blocking on warnings, requiring tests — rather than
relying on the prompt to remember it.

# Enabling the optional hooks {#coding_agents_llms_settings_local}

`.claude/settings.local.json` holds per-user, machine-local settings that merge
over the shared `.claude/settings.json`. To turn on the snapshot/warn/verify
hooks, create `.claude/settings.local.json` with:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Update|Write|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/SnapshotState.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/WarnCapture.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/StopVerify.sh",
            "timeout": 30000
          }
        ]
      }
    ]
  }
}
```

Each `hooks` block wires one of the optional hooks from
\ref coding_agents_llms_hooks to its lifecycle event (the `Stop` hook is given
a 30-second `timeout`).

# Effective agentic workflow {#coding_agents_llms_workflow}

A few practices make agentic coding in SpECTRE productive:

- **Use Plan mode for any non-trivial change.** Have the agent research the
  code and produce a written plan first, then read it carefully and iterate on
  it until it is correct *before* any edits are made. Getting the plan right is
  where most of the value lies — a wrong plan executed quickly just wastes a
  build.
- **Use normal ("no" / ask) mode to explore.** It is ideal for discussing
  design, asking the agent to trace code paths, and understanding the
  repository without making changes.
- **Reserve accept-edits and auto modes for small, well-scoped adjustments.**
  For larger work, keep a human in the loop.
- **Let the hooks and skills do the mechanical enforcement** (formatting,
  warnings, tests, reviews) and keep your `CLAUDE.md`/`AGENTS.md` tight and
  high-signal.
- **You are responsible** the agent may produce the code, but you are still
  responsible for it. You must review it carefully, think about what it is
  doing, consider tradeoffs, etc. Before submitting a PR, make sure to review
  agent-generated code yourself first. Don't waste other people's time.
