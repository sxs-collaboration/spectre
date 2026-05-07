# SpECTRE Code Review

**Arguments**: $ARGUMENTS
**Current branch**: !`git branch --show-current`

Perform a thorough code review of SpECTRE changes. Follow every step below
precisely. The full SpECTRE code rules reference (provide to review agents):
!`tail -n +5 .claude/rules/Cxx.md`

## Step 1: Parse Arguments & Acquire Diff

Parse `$ARGUMENTS`:
- **Number present** (e.g. `1234`): PR mode. Fetch with:
  `gh pr diff <N> --repo sxs-collaboration/spectre` and
  `gh pr view <N> --repo sxs-collaboration/spectre --json
  title,body,headRefName,baseRefName`
- **No number**: Local mode. Use `git diff develop...HEAD` for committed
  changes, plus `git diff` and `git diff --cached` for uncommitted changes. If
  no diff, fall back to `git diff HEAD~1`.
- **"clang-tidy" present**: Enable the clang-tidy step.
- **"coverage" present**: Enable the targeted code coverage step.

Save the diff. Extract the list of changed files categorized by type (C++
`.cpp/.hpp/.tpp`, Python `.py`, CMake `CMakeLists.txt`, other).

Create a task list tracking all review steps.

Read `references/reviewer-feedback-checklist.md`. Use it as an additional
checklist throughout the review and provide it to review agents in Step 4.

## Step 2: Formatting Checks (run in parallel)

### C++ (clang-format)
For each changed C++ file:
1. Parse diff hunk headers (`@@ +START,COUNT @@`) to get changed line ranges
2. Expand each range by +/-4 lines (clamped to file bounds)
3. Run: `clang-format -style=file --lines=START:END [--lines=...] FILEPATH`
4. Diff against original. Collect any formatting differences.

### Python (black + isort)
For each changed `.py` file (excluding `external/`):
```
black --check --diff FILEPATH
isort --check-only --diff FILEPATH
```

## Step 3: CI Pre-Checks on Changed Files (run in parallel)

Check each changed file for issues that SpECTRE CI (`tools/FileTestDefs.sh`,
`tools/CheckFiles.sh`) will flag. Only check lines/patterns introduced in the
diff, not pre-existing issues.

**All files**: lines >80 chars (excluding URLs, NOLINT, #include, \snippet,
\image, \link, import); missing MIT license header; no final newline; tabs;
trailing whitespace; carriage returns

**C++ files (.cpp/.hpp/.tpp)**:
- Missing `#pragma once` in headers
- `#include <iostream>` -> use `Parallel/Printf/Printf.hpp`
- `#include <lrtslock.h>` -> use `<converse.h>`
- `std::enable_if` -> use `requires`
- `namespace _details` -> use `_detail`
- `struct TD;` / `class TD;` (debug artifacts)
- `.ckLocal()` -> `Parallel::local()`
- `.ckLocalBranch()` -> `Parallel::local_branch()`
- `return Py_None;` -> `Py_RETURN_NONE`
- Text after `/*!` on same line
- `Ls` abbreviation -> use `List`
- Doxygen (`///` or `/*!`) in `.cpp` files (only use `//` comments in cpp)
- Top-level `const` on value parameters in function declarations (`.hpp`) ->
  remove `const` from the declaration (keep in definition)
- TODO/FIXME comments (not allowed)

**Test files**:
- `TEST_CASE` instead of `SPECTRE_TEST_CASE`
- `Approx(` instead of `approx`

**CMake**: New C++ files in a directory must be listed in that directory's
CMakeLists.txt; removed files must be removed; entries should be alphabetical.

**LLM Comments**: Identify comments that seem like notes from a coding agent
during its thinking process.

### Include Order (C++ files in diff)
Verify:
1. (Tests) `"Framework/TestingFramework.hpp"` first, then blank line
2. (`.cpp` with `.hpp`) Corresponding `.hpp`, then blank line
3. STL/external `<headers>` alphabetical
4. Blank line
5. SpECTRE `"headers"` alphabetical

### Commit Messages (local mode only)
Check no commit starts with (case-insensitive): fixup, wip, fixme, deleteme,
rebaseme, testing, rebase.
Check co-author trailers are in commit messages, separated from the body by a
blank line, so GitHub recognizes them.

## Step 4: Code Review (2 Parallel Agents)

Launch 2 parallel agents. Provide each with the full diff and the
SpECTRE code rules reference shown above, plus
`references/reviewer-feedback-checklist.md`.

### Agent A: Style, Patterns & Idioms
Instructions for the agent:
- Check the diff against every rule in **Banned Patterns** and **Style Rules**
- Check for **Prefer-Library Patterns** (manual tensor loops that should use
  EagerMath)
- When you spot a suspicious pattern NOT in the checklist (e.g., a manual matrix
  operation, a loop that looks like it reimplements an existing utility), use
  `grep -r` in `src/DataStructures/Tensor/EagerMath/`, `src/DataStructures/`,
  `src/NumericalAlgorithms/`, or `src/Utilities/` to find an existing utility
- Check the recurring reviewer-feedback checklist, especially commit hygiene,
  formatting churn, CMake ordering, includes, existing local helpers, and API
  naming/documentation
- Only flag issues in lines that the diff introduces (not pre-existing code)
- For each finding: `file:line`, severity (`critical`/`important`/`suggestion`),
  explanation

### Agent B: Bugs, Logic, Tests & Documentation
Instructions for the agent:
- Read each changed file in full (not just the diff) to understand surrounding
  context
- Look for: logic errors, off-by-one, uninitialized variables, NaN handling,
  race conditions, incorrect template instantiations, virtual inheritance issues
  (most-derived must init virtual bases)
- Check that new/changed public API in `.hpp` files has Doxygen documentation
- Check that new source files have corresponding tests (`src/Foo/Bar.hpp` ->
  `tests/Unit/Foo/Test_Bar.cpp`)
- Check that new `.cpp`/`.hpp` files are listed in their `CMakeLists.txt`
- Check the recurring reviewer-feedback checklist, especially weak tests,
  tolerance relaxations, missing ASSERT/ERROR tests, unnecessary allocations,
  copies, expensive recomputation, NaN handling, and unclear numerical docs
- Only flag issues introduced by the diff
- For each finding: `file:line`, severity, explanation
- Check for potentially problematic floating point math like:
  - Catastrophic cancellation: subtraction of nearly-equal values
    (`a - b` where `a ~ b`).
  - Division by near-zero: a denominator that approaches zero for certain
    inputs (e.g., near coordinate boundaries, poles, or special points).
  - Naive summation of many terms: summing a long list of floating-point
    numbers accumulates round-off.
  - Unstable polynomial evaluation: monomial form
    (`a*x^3 + b*x^2 + c*x + d`), instead use Horner's method,
    `evaluate_polynomial()` from `src/Utilities/Math.hpp`
  - Numerically unstable quadratic formula: the standard formula
    `(-b +/- sqrt(b^2 - 4ac)) / 2a` loses precision.
  - Accumulation of terms with widely varying magnitudes: adding a tiny
    correction to a large value drops the correction entirely.
  - Make sure the code always uses `atan2` instead of `atan`, `log1p(x)` instead
    of `log(1+x)`, `expm1(x)` instead of `exp(x)-1`, `hypot(x,y)` instead of
    `sqrt(x*x+y*y)`.

## Step 5: clang-tidy (if requested, run in parallel)

If "clang-tidy" was in arguments:
1. Check for `build/compile_commands.json`. If missing, report that clang-tidy
   requires a configured build directory and skip.
2. For each changed `.cpp` file: `clang-tidy -p build/ FILEPATH 2>&1`
3. Filter output to only warnings on lines in the diff.

## Step 6: Code Coverage (if requested, run in parallel)

If "coverage" was in arguments:
Read `references/coverage-steps.md` and follow those instructions exactly.

## Step 7: Self-Review Prune

Combine all findings from steps 2-6. Review each finding and REMOVE only clear
non-issues:
- False positives (pattern match that isn't the actual flagged issue)
- Pre-existing issues not introduced by this diff
- Issues suppressed by `// NOLINT(...)` comments
- Exact duplicates between agents or between agents and formatting/CI checks

Assign each remaining finding a confidence score (0-100). Remove findings
below 50. Keep all findings scoring 50 or above -- err on the side of including
borderline issues rather than missing real ones.

## Step 8: Lightweight Model Critique

Spawn an agent using the cheapest available model (Claude Code: `haiku`;
Codex: `gpt-5.4-mini`). Provide it with:
- The SpECTRE code rules reference (from Step 1)
- The list of pruned findings (with scores)
- A summary of what the diff does

Ask the critique agent to:
1. Score each finding 0-100 for "is this a real, actionable issue?"
2. Flag any remaining false positives with reasoning
3. Note if important issues seem to be missing
4. Return scores and feedback

After receiving the critique agent's feedback:
- Remove findings scored < 40
- Downgrade severity (e.g., important -> suggestion) for findings scored
  40-60
- Consider adding issues the critique agent suggested (verify them first)

## Step 9: Final Report

Read `references/report-template.md` and present the report in that format.
