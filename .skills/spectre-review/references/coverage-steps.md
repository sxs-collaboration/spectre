# Code Coverage Steps

These instructions are followed by the orchestrator when "coverage" was in
the review arguments. Execute all sub-steps below exactly as written.

### 6a. Check out PR branch (PR mode only)

Coverage must be measured against the actual PR code, not the current working
tree. Before building or running any tests:

1. Record the current branch and stash any uncommitted changes so they can be
   restored afterwards:
   ```bash
   ORIGINAL_BRANCH=$(git branch --show-current)
   STASH_RESULT=$(git stash push --include-untracked -m "coverage-stash" 2>&1)
   STASHED=$([[ "$STASH_RESULT" == *"Saved"* ]] && echo yes || echo no)
   ```
2. Attempt to check out the PR branch using the GitHub CLI:
   ```bash
   gh pr checkout <N> --repo sxs-collaboration/spectre 2>&1
   ```
   If that fails (e.g. SSH key / remote mismatch), fall back to:
   ```bash
   git fetch upstream pull/<N>/head:pr-<N> 2>&1
   git checkout pr-<N> 2>&1
   ```
3. **Handle checkout conflicts explicitly.** If checkout fails with any error
   (e.g. "Your local changes to the following files would be overwritten",
   "untracked working tree files would be overwritten by checkout", or "Please
   commit your changes or stash them"), do NOT attempt to force-checkout.
   Instead, abort coverage and report clearly:
   > "Coverage analysis skipped: checking out PR branch '<headRefName>' failed
   > because the following local files conflict with the PR:
   >   <list conflicting files from the git error message>
   > Please resolve the conflict manually (commit, stash, or delete those files)
   > and re-run the review with 'coverage'."
   Then restore the original branch and pop the stash (step 5) and skip all
   remaining coverage steps.
4. Verify the checkout succeeded by comparing HEAD to the expected PR commit:
   ```bash
   git log --oneline -1
   gh pr view <N> --repo sxs-collaboration/spectre \
       --json headRefOid -q .headRefOid
   ```
   The two commit hashes should match (or the local log should show the PR
   branch name). If they differ, report: "Coverage skipped: checkout succeeded
   but HEAD does not match PR head — local branch may be stale."
5. **Restore step (must run after all coverage steps, or on any failure after
   this point):** at the end of Step 6f, restore the original working state:
   ```bash
   git checkout "$ORIGINAL_BRANCH"
   [[ "$STASHED" == "yes" ]] && git stash pop
   ```

### 6b. Prerequisites & Build Verification

1. Check `build/compile_commands.json` exists. If missing, skip with: "Coverage
   requires a configured build directory."
2. Check `build/CMakeCache.txt` for `COVERAGE:BOOL=ON`. If not found, skip with:
   "Build not compiled with -DCOVERAGE=ON. Rebuild with
   `cmake -DCOVERAGE=ON ..` to enable coverage."
3. Check `lcov` is available.
4. **Verify the gcov wrapper is functional.** For Clang builds,
   `build/llvm-gcov.sh` may contain `LLVM_COV_BIN-NOTFOUND` if `llvm-cov` was
   not found at CMake configure time. Check:
   ```bash
   grep "NOTFOUND" build/llvm-gcov.sh
   ```
   If found, locate the correct binary and fix the wrapper in-place:
   ```bash
   LLVM_VER=$(clang++ --version | grep -oP '\d+' | head -1)
   LLVM_COV=$(which llvm-cov-${LLVM_VER} 2>/dev/null || which llvm-cov)
   printf '#!/bin/bash\nexec %s gcov "$@"\n' "$LLVM_COV" > build/llvm-gcov.sh
   chmod +x build/llvm-gcov.sh
   ```
5. Determine the gcov tool and **always use its absolute path** to avoid lcov
   failing with "No such file or directory" when it resolves relative paths
   from a different working directory:
   - Clang: `GCOV=$(realpath build/llvm-gcov.sh)`
   - GNU: `GCOV=$(which gcov)`
6. **Verify objects are instrumented.** Pick one `.o` file for a changed source
   and check it has coverage symbols:
   ```bash
   nm <path/to/the_file.cpp.o> 2>/dev/null | grep -c "__llvm_gcov_ctr\|__gcov_"
   ```
   If the count is 0, objects were compiled before coverage was enabled —
   rebuild the relevant test targets:
   ```bash
   cmake --build build --target <TestTarget> -- -j$(nproc)
   ```
   Re-check instrumentation after the rebuild. If still 0, skip coverage with:
   "Objects lack coverage instrumentation even after rebuild. Check that the
   build was configured with -DCOVERAGE=ON before compiling."

### 6c. Map changed files to test targets
For each changed C++ file under `src/`:
1. Find the corresponding test directory: `src/A/B/C.hpp` -> `tests/Unit/A/B/`
2. Walk up directories reading `CMakeLists.txt` files for `set(LIBRARY
   "Test_...")` to find the test binary name. Test naming is NOT a simple
   transform (e.g., `Test_DomainCreators`, `Test_EllipticDG`, `Test_DgSubcell`),
   so CMakeLists.txt parsing is required.
3. Verify each binary exists in `build/bin/`. If not, build it: `cmake --build
   build --target <Target> -- -j$(nproc)`
4. If >5 unique test targets, ask the user before proceeding.

### 6d. Run tests and capture coverage
For each test target (using `$GCOV` absolute path from Step 6b):
```bash
lcov --gcov-tool $GCOV --directory build/ --zerocounters
build/bin/<TestTarget>
lcov --gcov-tool $GCOV --capture --rc lcov_branch_coverage=0 \
  --directory build/ --output-file /tmp/coverage_<TestTarget>.info
```
After the lcov capture, verify `.gcda` files were actually produced:
```bash
find build/ -name "*.gcda" | grep -q .
```
If no `.gcda` files are found, report:
> "No coverage data files (.gcda) generated. Possible causes: (1) objects were
> not compiled with --coverage (rebuild required after confirming
> -DCOVERAGE=ON), (2) test binary exited before writing data, (3) build
> directory path mismatch between compile and run."
Then skip remaining coverage steps (but still run Step 6f to restore the
branch).

### 6e. Filter to changed lines only
1. Merge .info files: `lcov --add-tracefile ... --output-file
   /tmp/coverage_merged.info`
2. Extract only changed source files: `lcov --extract /tmp/coverage_merged.info
   '<abs-path-to-file>' ... -o /tmp/coverage_filtered.info`
3. Parse `DA:<line>,<count>` entries from the filtered .info file
4. Cross-reference with diff hunk headers: only report lines that are both NEW
   in the diff AND have execution_count == 0. **Use new-file line numbers from
   the diff** — after checking out the PR branch these match the files on disk
   exactly. Do not use base-branch line numbers, which are offset from the PR.
5. Group uncovered lines into contiguous ranges

### 6f. Cleanup
Remove temporary .info files to avoid polluting subsequent builds:
```bash
rm -f /tmp/coverage_*.info /tmp/coverage_merged.info /tmp/coverage_filtered.info
```
Then restore the original branch and unstash local changes (PR mode only):
```bash
git checkout "$ORIGINAL_BRANCH"
[[ "$STASHED" == "yes" ]] && git stash pop
```
