# Fixing a Random Test Failure

You have a random test failure issue. The script output above contains
the test name and a list of `file:line` locations with their failing seeds.

## Background

`MAKE_GENERATOR(gen)` (defined in `tests/Unit/Framework/TestHelpers.hpp`)
creates a `std::mt19937` with a random seed. `MAKE_GENERATOR(gen, SEED)`
uses the provided constant instead, making the test deterministic. On
failure, prints `Seed is: N from file:line` so the failure can be replayed.

## Strategy A -- Seed from a test file

Use when the `file:line` points to a test file (not
`CheckWithRandomValues.hpp`).

1. Open the file at the indicated line (line numbers may have shifted -- search
   nearby for `MAKE_GENERATOR`).
2. Change `MAKE_GENERATOR(gen);` to `MAKE_GENERATOR(gen, SEED);` using the
   first seed listed for that location.
3. Build and run to reproduce the failure (see Workflow below).
4. The failure is almost always a numerical tolerance issue. Increase the
   comparison tolerance **locally** (only on the specific failing check):
   - `Approx custom_approx = Approx::custom().epsilon(TOL).scale(1.0);`
   - For scalars: `CHECK(expected == custom_approx(computed));`
   - For iterables:
     `CHECK_ITERABLE_CUSTOM_APPROX(expected, computed, custom_approx);`
   - Start with `1.0e-14` and multiply by 10 until the test passes. Keep the
     tolerance as small as possible.
5. Remove the seed so it returns to `MAKE_GENERATOR(gen);`.
6. Verify with `ctest --repeat-until-fail 1000 -R "TEST_NAME"` in the build
   directory. If any iterations fail, fix those too. Repeat until all 1000 pass.

## Strategy B -- Seed from CheckWithRandomValues.hpp

Use when the `file:line` points to `Framework/CheckWithRandomValues.hpp`.

`check_with_random_values()` already accepts `epsilon` and `seed` parameters
(in `tests/Unit/Framework/CheckWithRandomValues.hpp`). The fix is:

1. Find the calling test's `check_with_random_values` invocation.
2. Pass the failing seed as the last argument and adjust `epsilon` as needed.
   For example, if the original call is:
   ```cpp
   pypp::check_with_random_values<1>(
       &Foo::bar<DataType>, klass, "Module", "function",
       {{{-10.0, 10.0}}}, member_vars, used_for_size);
   ```
   add `epsilon` (default is `1.0e-12`) and the seed:
   ```cpp
   pypp::check_with_random_values<1>(
       &Foo::bar<DataType>, klass, "Module", "function",
       {{{-10.0, 10.0}}}, member_vars, used_for_size,
       1.0e-12, FAILING_SEED);
   ```
3. Build and run to reproduce the failure.
4. Increase the `epsilon` parameter until the test passes (start at `1.0e-12`,
   multiply by 10). Keep as small as possible.
5. Remove the explicit seed argument after the fix.
6. Verify with `ctest --repeat-until-fail 1000 -R "TEST_NAME"`.

## Workflow

For each `file:line` + seed(s) entry from the script output:

1. **Set seed** per Strategy A or B above.
2. **Find build target**:
   ```
   cd build && ctest --show-only=json-v1 -R "TEST_NAME"
   ```
   Parse the JSON output for the binary path (first element of `"command"`).
3. **Build**: `ninja -C build <binary_name>`
4. **Run**: `build/bin/<binary_name> "TEST_NAME"`
5. **If test does NOT fail**: STOP and report to the user that the failure
   cannot be reproduced (code may have changed since the issue was filed).
6. **Investigate the root cause** before applying a tolerance fix. Read the
   error output to identify the failing comparison (e.g. which function's
   result is being compared). Then trace through the call chain of the
   function under test to determine the source of the numerical error:
   - **Iterative solver / root-find**: Look for convergence tolerances,
     iteration counts, or precision parameters that could be tightened.
   - **Adjustable numerical parameter**: Look for step sizes, interpolation
     orders, or quadrature rules that could be refined.
   - **Inherent floating-point accumulation**: If the computation is fully
     analytical (no iteration, no adjustable precision knobs), the error is
     unavoidable in double precision.

   If you find an adjustable parameter, fix that instead of loosening the
   test tolerance. If the error is inherent, proceed with the tolerance fix
   and document the root cause in a C++ comment next to the tolerance change.
7. **Apply fix**: verify the test passes with the seed, then remove the seed
   and run `ctest --repeat-until-fail 1000 -R "TEST_NAME"`.
8. If multiple `file:line` entries exist, repeat for each one.
9. **Report a summary** of all changes at the end: list every file, line, and
   what was changed (tolerance epsilon, solver parameter, etc.) so the user
   can assess whether any changes are too large or need further review.

## Important notes

- The build directory is `build/` under the repository root.
- Do NOT modify global `approx` in `tests/Unit/Framework/TestingFramework.hpp`.
- Always remove explicit seeds before committing -- the seed is only for
  reproducing the failure, not for the final code.
- If multiple distinct test files have seeds, investigate each independently.
