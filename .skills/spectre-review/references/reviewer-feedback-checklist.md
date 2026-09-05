# Recurring SpECTRE Reviewer Feedback Checklist

Use this checklist when reviewing local changes, PRs, or PR readiness. Only
flag issues introduced by the diff, unless the user explicitly asks for a
broader audit.

## Commit and PR Hygiene

- Co-author trailers belong in commit messages, not only PR descriptions.
- Commit messages should have a subject, a blank line before any body text, and
  a blank line before trailers so GitHub recognizes `Co-Authored-by` trailers.
- Flag obvious `fixup!`, `WIP`, or review-fix commits in PR-ready work.
- If commits introduce a bad pattern and later fix it, suggest squashing before
  review.

## Formatting and Includes

- Avoid whole-file formatting churn. Prefer `git clang-format` or changed-line
  formatting.
- Watch for stale clang-format output, especially around `requires` clauses.
- CMake source/header lists must stay alphabetized.
- New symbol uses should have direct includes, not rely on transitive includes.
- Keep include blocks in SpECTRE order. For CLI Python modules, delayed imports
  may be intentional to avoid startup cost.

## Tests

- Unit tests should live in `tests/Unit` alongside the module under test.
- Keep integration or regression scenarios in dedicated subdirectories.
- Name tests after the component and behavior being validated.
- Treat tolerance relaxations as suspicious. Prefer finding the numerical cause
  before raising tolerances.
- Tests should verify correctness, not just repeat the implementation or check
  sizes/counts.
- New functionality should have focused test coverage.
- Prefer identity/metamorphic tests when possible, such as Cartoon vs
  non-Cartoon agreement on axes.
- Add `CHECK_THROWS_WITH` coverage for new `ASSERT`/`ERROR` paths when
  practical.
- Use `signaling_NaN()` for data that should not be used.
- Avoid multiple active `MAKE_GENERATOR` instances; pass generators to helpers.
- Prefer one `SPECTRE_TEST_CASE` calling anonymous-namespace helpers over many
  small `SPECTRE_TEST_CASE`s.
- Increase test timeouts sparingly and justify long-running cases.
- Test relevant data types, including `double`, when behavior differs from
  `DataVector`.

## Performance and Allocations

- In numerical kernels and reusable utilities, look for allocations in loops or
  repeated calls.
- Prefer `Variables`, explicit temp buffers, or cached matrices for repeated
  tensor/data operations.
- Avoid unnecessary copies; use `const auto&` when a copy is not intended.
- Do not zero-initialize data that is completely overwritten. Debug NaN
  initialization is often better for catching errors.
- Cache expensive transform/filter matrices instead of rebuilding them on each
  call.
- Be careful with Blaze/DataVector expression templates: `auto` can bind lazy
  expressions when a concrete `DataType` is intended.

## Numerical Robustness

- Check for cancellation, especially expressions like `1 - W`.
- Check denominators near coordinate boundaries, poles, origins, and degenerate
  intervals.
- Prefer stable library functions: `atan2`, `log1p`, `expm1`, `hypot`, Horner
  evaluation or `evaluate_polynomial()`.
- If a test fails near a singular/degenerate case, consider whether `src` has a
  real bug before relaxing the test.
- Use sufficiently resolved test grids so tolerances can detect real errors.

## Documentation and API Shape

- New public APIs need Doxygen that explains assumptions, valid domains,
  restrictions, data layout, and component interpretation.
- Mathematical/numerical code should include equations or citations when the
  implementation is not obvious.
- For tensor transforms, document whether data is Cartesian, spherical,
  spin-weighted, nodal, modal, strided, or radially batched.
- Prefer established local terms such as `stride` or `radial_extents` over vague
  names like `number_of_offsets`.

## Existing Patterns

- Before accepting a new helper, search for local precedent in
  `src/DataStructures/Tensor/EagerMath/`, `src/NumericalAlgorithms/`,
  `src/PointwiseFunctions/`, and `src/Utilities/`.
- Prefer existing helpers and patterns such as `to_different_frame`,
  `apply_tensor_ylm_filter`, `ActionTesting::invoke_queued_simple_action`,
  EagerMath functions, and existing domain creator designs.
