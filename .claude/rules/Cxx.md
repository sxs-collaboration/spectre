---
triggers:
  - glob: "**/*.{hpp,cpp,tpp}"
---

# SpECTRE Code Rules Reference

## Banned Patterns
- `#include <iostream>` -> `Parallel/Printf/Printf.hpp` with `Parallel::printf`
- `#include <lrtslock.h>` -> `#include <converse.h>`
- `std::enable_if` -> C++20 `requires` clause
- `||` `&&` `!` (logical operators) -> `or` `and` `not` (C++ alternative
  tokens). Note: `!=` is fine.
- `.ckLocal()` -> `Parallel::local(proxy)`
- `.ckLocalBranch()` -> `Parallel::local_branch(proxy)`
- `return Py_None;` -> `Py_RETURN_NONE`
- `TEST_CASE(` -> `SPECTRE_TEST_CASE(`
- `Approx(` in tests -> `approx` from `Framework/TestingFramework.hpp`
- `namespace _details` / `namespace details` -> `_detail` / `detail`
- `boost::optional` / `boost/optional.hpp` -> `std::optional`
- `mutable` member variables -> avoid (race conditions in parallel); discuss
  with core devs
- `struct TD;` / `class TD;` -> remove (debug artifacts)
- `Ls` as abbreviation -> use `List`
- `int` for container sizes/indices -> `size_t`
- `noexcept` -> don't add unless overriding a signature that requires it

## Prefer-Library Patterns
When you see manual loops over Tensor indices, suggest the existing utility:
- Dot product -> `DataStructures/Tensor/EagerMath/DotProduct.hpp`
- Cross product -> `EagerMath/CrossProduct.hpp`
- Trace -> `EagerMath/Trace.hpp`
- Determinant -> `EagerMath/Determinant.hpp` or `DeterminantAndInverse.hpp`
- Magnitude/norm -> `EagerMath/Magnitude.hpp` or `Norms.hpp`
- Outer product -> `EagerMath/OuterProduct.hpp`
- Raise/lower index -> `EagerMath/RaiseOrLowerIndex.hpp`
- Gram-Schmidt -> `EagerMath/GramSchmidtOrthonormalize.hpp`
- Frame transform -> `EagerMath/FrameTransform.hpp`
- Cartesian<->Spherical -> `EagerMath/CartesianToSpherical.hpp`

If you spot a tensor loop not covered above, grep
`src/DataStructures/Tensor/EagerMath/` and `src/NumericalAlgorithms/` for an
existing utility before suggesting the author write their own. Encourage use
of tensor expressions `src/DataStructures/Tensor/Expressions/`.

For general relativity, generalized harmonic, CCZ4, apparent horizons, and
GRMHD, many manipulations like shift from spacetime metric, lapse from spacetime
metric, derivative of shift, etc. are implemented as functions in
`src/PointwiseFunctions/GeneralRelativity` or `src/PointwiseFunctions/Hydro`.

## Style Rules
- **Naming**: CamelCase for classes, template params, files, dirs. snake_case
  for functions, variables. SCREAMING_SNAKE_CASE for macros. Trailing `_` on
  private members. Unused params: `/*name*/`.
- **Names**: Use full, descriptive names (e.g., `block_id` not `blk_id`).
- **Almost always `auto`** except expression templates (e.g. `DataVector`)
- **Braces on all loops and if/else** (no braceless one-liners)
- **Return by value** preferred. Mutable out-params: `gsl::not_null<T*>` (listed
  first in arg list).
- **`std::move`** into members from by-value constructor parameters
- **`get<a,b>(tensor)`** when indices are compile-time known
- **Explicit double literals**: `2.0` or `2.`, never bare `2` in floating-point
  context
- **`#pragma once`** for header guards (not `#ifndef`)
- **`override`** keyword on all virtual overrides
- **`const`** on all immutable objects
- **No top-level `const` on value parameters in declarations**: In `.hpp`
  forward declarations, don't mark value parameters (including pointer-by-value)
  as `const`. `const` is fine in the corresponding definition. Example:
  `void foo(int x, double* ptr);` (declaration) vs
  `void foo(const int x, double* const ptr) { ... }` (definition).
- **Templates**: prefer definitions in `.cpp` with explicit instantiations
  (`GENERATE_EXPLICIT_INSTANTIATIONS`)
- **Internal namespaces**: `LibraryOrFile_detail` (e.g. `Tensor_detail`)
- **Macros**: avoid if possible; use `constexpr` or templates. Macro vars: add
  suffix to avoid collisions.
- **Error messages**: descriptive, include runtime values. Bad: "Size
  mismatch". Good: "The number of grid points in matrix 'F' (N) != determinant
  grid points (M)."
- **Header order**: (1) TestingFramework.hpp + blank (tests), (2) corresponding
  .hpp + blank (.cpp files), (3) STL/external `<>` alphabetical, (4) blank, (5)
  SpECTRE `""` alphabetical
- **Doxygen**: required for all public API in `.hpp`. Use `///` or `/*!`. NO
  doxygen in `.cpp`. Use `\f$...\f$` for inline math, `align` env (not
  `eqnarray`) for multi-line. Blank doxygen line before/after out-of-line
  equations.
- **CMake lists**: alphabetical. Use `${LIBRARY}` variable, not hardcoded names.

## Test Requirements
- Files: `tests/Unit/<mirrors_src>/Test_<SourceFile>.cpp`
- Include order: `"Framework/TestingFramework.hpp"`, blank line,
  system/external includes, blank line, spectre includes.
- All helper classes/functions in anonymous `namespace {}`
- Test macro: `SPECTRE_TEST_CASE("Unit.Category.Name", "[Unit][Category]")`
- Floating-point: `CHECK_ITERABLE_APPROX` (not `Approx`)
- Completion: < 5 seconds (prefer < 0.5s)
- Random values: `MAKE_GENERATOR(gen)`, test 10^4 times for tolerance
- Error tests: `CHECK_THROWS_WITH` inside `#ifdef SPECTRE_DEBUG`
- Pointwise functions: test with analytic solution AND random-value comparison
  via `pypp::check_with_random_values()`
- Use meromorphic tests: like `sin^2(x)+cos^2(x)=1` or that a spacetime vector
  in general relativity that should be null is actually null. I.e, test
  identities.
- Name tests after the component and behavior (e.g.,
  `Test_ApparentHorizonFinder`).
- Prefer a single `SPECTRE_TEST_CASE` that calls several anonymous-namespace
  helper functions over many small `SPECTRE_TEST_CASE`s.
- Increase test timeouts sparingly. If necessary, use `// [[TimeOut, SECONDS]]`
  on the line before `SPECTRE_TEST_CASE`. Default timeout is 2 seconds.
