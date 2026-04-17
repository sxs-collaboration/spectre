# Fixing a Random Test Failure

You have a random test failure issue. The script output above contains
the test name and a list of `file:line` locations with their failing seeds.

Random failures are almost always caused by a **numerical bug in the source
code**, not by overly tight test tolerances. Your job is to find and fix
the source-level bug. Increasing a test tolerance is a last resort, gated
behind mandatory investigation.

## 1. Background

`MAKE_GENERATOR(gen)` (defined in `tests/Unit/Framework/TestHelpers.hpp`)
creates a `std::mt19937` with a random seed. `MAKE_GENERATOR(gen, SEED)`
uses the provided constant instead, making the test deterministic. On
failure, prints `Seed is: N from file:line` so the failure can be replayed.

### Seeding a test file

When the `file:line` points to a test file (not `CheckWithRandomValues.hpp`):
open the file at the indicated line (line numbers may have shifted -- search
nearby for `MAKE_GENERATOR`) and change `MAKE_GENERATOR(gen);` to
`MAKE_GENERATOR(gen, SEED);` using the first seed listed for that location.

### Seeding CheckWithRandomValues

When the `file:line` points to `Framework/CheckWithRandomValues.hpp`:
`check_with_random_values()` accepts `epsilon` and `seed` parameters. Find
the calling test's `check_with_random_values` invocation and pass the failing
seed as the last argument. For example, if the original call is:
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

## 2. Reproduce the Failure

For each `file:line` + seed entry from the script output:

1. **Set the seed** per Section 1 above.
2. **Find the build target**:
   ```
   cd build && ctest --show-only=json-v1 -R "TEST_NAME"
   ```
   Parse the JSON output for the binary path (first element of `"command"`).
3. **Build**: `ninja -C build <binary_name>`
4. **Run**: `build/bin/<binary_name> "TEST_NAME"`
5. **If the test does NOT fail**: STOP and report to the user that the failure
   cannot be reproduced (code may have changed since the issue was filed).

## 3. Investigate the Root Cause (REQUIRED)

This is the most important section. You MUST complete this investigation
before considering any tolerance change.

### 3a. Trace the call chain

1. Read the error output to identify the **exact failing comparison** -- which
   values were compared, what the expected vs. actual values were, and what the
   discrepancy is.
2. Identify which **function under test** produces the incorrect value.
3. **Read the source code** of that function. Trace through its arithmetic step
   by step. Identify the specific operation (e.g., "the subtraction on line 438
   of Wedge.cpp") that amplifies floating-point error.
4. If the function calls other functions, trace into those as well. Follow the
   data flow until you reach the arithmetic that is numerically unstable.

You MUST identify a specific line and operation in the source code. "The
computation accumulates floating-point error" is not sufficient -- you need to
say *where* and *why*.

### 3b. Numerical patterns checklist

Search the source code in the call chain for these common floating-point
pitfalls. For each pattern found, attempt the listed fix:

- **Catastrophic cancellation**: subtraction of nearly-equal values
  (`a - b` where `a ~ b`). Fix: rearrange the algebra to avoid the subtraction,
  use compensated forms, or factor out common terms.

- **Division by near-zero**: a denominator that approaches zero for certain
  inputs (e.g., near coordinate boundaries, poles, or special points). Fix:
  reformulate to multiply instead of divide, combine fractions to cancel the
  small denominator, or restructure the expression analytically.

- **`atan` instead of `atan2`**: `atan(y/x)` loses quadrant information and
  divides by near-zero when `x ~ 0`. Fix: replace with `atan2(y, x)`.

- **Missing precision-preserving library functions**: `log(1+x)` loses
  precision when `x ~ 0`; `exp(x)-1` loses precision when `x ~ 0`;
  `sqrt(x*x + y*y)` can overflow for large values. Fix: use `log1p(x)`,
  `expm1(x)`, and `hypot(x, y)`.

- **Naive summation of many terms**: summing a long list of floating-point
  numbers accumulates round-off. Fix: use Kahan (compensated) summation, or
  sort terms by magnitude before summing.

- **Unstable polynomial evaluation**: monomial form
  (`a*x^3 + b*x^2 + c*x + d`) accumulates error for large `x`. Fix: use
  Horner's method (`((a*x + b)*x + c)*x + d`).

- **Numerically unstable quadratic formula**: the standard formula
  `(-b +/- sqrt(b^2 - 4ac)) / 2a` loses precision when `b^2 >> 4ac` due to
  catastrophic cancellation in the numerator. Fix: use the numerically stable
  form (compute one root via the standard formula, the other via
  `c / (a * first_root)`).

- **Accumulation of terms with widely varying magnitudes**: adding a tiny
  correction to a large value drops the correction entirely. Fix: reorder
  operations so values of similar magnitude are combined first, or use
  compensated summation.

### 3c. Attempt a source-level fix

Based on the pattern(s) you identified in 3a and 3b:

1. **Implement a fix in the source code** (not the test). For example:
   rearrange an expression to avoid cancellation, replace a division with a
   multiplication, switch `atan` to `atan2`, etc.
2. **Build and run with the failing seed** to verify the fix resolves the
   failure.
3. If the fix works, skip Section 4 entirely and go to Section 5 (Verify).
4. If the fix does not fully resolve the issue, document what you tried and
   why it was insufficient, then continue investigating. Only proceed to
   Section 4 after exhausting source-level options.

## 4. LAST RESORT -- Tolerance Adjustment

**MANDATORY GATE -- you may NOT adjust any tolerance until ALL of the
following are true:**

1. You have identified the full call chain from the test to the failing
   arithmetic operation.
2. You have named the specific source-code operation (file, line, expression)
   that produces the numerical error.
3. You have attempted at least one source-level fix and can explain why it
   did not work.
4. You have confirmed the error is inherent to the algorithm in double
   precision and cannot be eliminated by rearranging the arithmetic.

**If you cannot satisfy all four conditions, go back to Section 3.**

### Adjusting tolerance in a test file

- `Approx custom_approx = Approx::custom().epsilon(TOL).scale(1.0);`
- For scalars: `CHECK(expected == custom_approx(computed));`
- For iterables:
  `CHECK_ITERABLE_CUSTOM_APPROX(expected, computed, custom_approx);`
- Start with `1.0e-14` and multiply by 10 until the test passes. Keep the
  tolerance as small as possible.

### Adjusting tolerance in CheckWithRandomValues

Adjust the `epsilon` parameter in the `check_with_random_values` call.
Start at `1.0e-12` and multiply by 10. Keep as small as possible.

### Required documentation

Add a C++ comment next to the tolerance change explaining:
- What source-level operation causes the error
- Why a source-level fix is not feasible
- What magnitude of error is expected

## 5. Verify

1. **Remove the explicit seed** so the generator returns to
   `MAKE_GENERATOR(gen);` (or remove the seed argument from
   `check_with_random_values`).
2. **Run**: `ctest --repeat-until-fail 1000 -R "TEST_NAME"` in the build
   directory.
3. If any iterations fail, return to Section 3 and investigate the new failure.
   Repeat until all 1000 iterations pass.

## 6. Report Summary

After all `file:line` entries are resolved, report:
- Every file and line changed
- For **source-level fixes**: what the numerical issue was and how it was fixed
- For **tolerance adjustments**: the root cause, why a source fix wasn't
  feasible, and the new tolerance value

## Important Notes

- The build directory is `build/` under the repository root.
- Do NOT modify global `approx` in `tests/Unit/Framework/TestingFramework.hpp`.
- Always remove explicit seeds before committing -- the seed is only for
  reproducing the failure, not for the final code.
- If multiple distinct test files have seeds, investigate each independently.
