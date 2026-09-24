# SpECTRE PR Readiness

Prepare local SpECTRE changes for opening or marking a PR ready. This is a
pre-review workflow, not a substitute for human review.

## Step 1: Collect Scope

- Identify the base branch, normally `develop`.
- Gather `git diff develop...HEAD`, `git diff`, and `git diff --cached`.
- List changed files by type: C++, Python, CMake, docs, other.
- List local commits with subjects and full messages.

## Step 2: Run Mechanical Checks

- Run changed-line C++ formatting with `git clang-format` or the same
  changed-line procedure used by `spectre-review`.
- Run `black --check --diff` and `isort --check-only --diff` for changed Python
  files outside `external/`.
- Check CMake lists are updated and alphabetized.
- Check direct includes for newly used symbols.
- Check for whole-file formatting churn unrelated to the change.
- Run `git commit --dry-run` before finalizing unless the user asked not to
  commit or the tree is intentionally not commit-ready.

## Step 3: Review Commit Hygiene

- No `fixup!`, `WIP`, `testing`, `rebase`, or similar temporary commit subjects.
- Commit subjects should be short, descriptive, and imperative.
- Commit bodies should include context when behavior changes or dependencies are
  added.
- Reference issue numbers or external resources inline when available.
- Squash commits that introduce a mistake and later fix it, unless the user
  explicitly wants a multi-commit review history.
- Co-author trailers must be in commit messages, not only PR descriptions.
- Keep a blank line before trailer blocks so GitHub recognizes
  `Co-Authored-by` lines.

## Step 4: Run Recurring Feedback Checklist

Read `.skills/spectre-review/references/reviewer-feedback-checklist.md` and
apply it to the local diff. Focus on:

- weak tests, loose tolerances, missing failure-path tests
- unit-test placement in `tests/Unit`, test naming, focused coverage for new
  functionality, and use of anonymous-namespace helpers
- allocations, copies, expensive recomputation, expression-template pitfalls
- numerical robustness near poles, origins, degenerate intervals, and
  cancellation
- Doxygen assumptions, data layout, valid domains, equations, and citations
- reuse of existing SpECTRE helpers and local patterns

## Step 5: Build and Test

- Build targeted targets affected by the change when possible.
- Run focused `ctest --test-dir /home/vscode/work/builds/build -R <pattern>
  --output-on-failure`.
- If parallel tests time out, rerun timed-out tests serially before changing
  code.
- If changing docs, build `doc-check` when feasible.
- If clang-tidy is requested, use the repository clang-tidy hash script from
  `AGENTS.md` or run targeted clang-tidy against the configured build.
  To test all commits after, but not including, `HASH_BEFORE_COMMITS_TO_CHECK`,
  run:
  ```
  /home/vscode/work/builds/spectre-container/tools/ClangTidyHash.sh \
    /home/vscode/work/builds/build \
    /home/vscode/work/builds/spectre-container \
    HASH_BEFORE_COMMITS_TO_CHECK 8
  ```
  Typically `HASH_BEFORE_COMMITS_TO_CHECK` is the commit synced with
  `origin/develop`.

## Step 6: Report

Summarize blockers first, then important cleanup, optional polish, and testing
performed. Include exact commands run and any commands that could not be run.
For PR descriptions, summarize the motivation, user-visible effects, testing
performed, and relevant build/test commands. Reference issues or external
resources when useful, and attach visual artifacts or data files when they
clarify the change.
