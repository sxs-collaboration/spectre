// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/TypeTraits.hpp"

namespace {
template <bool HasTypeAlias>
struct AnalyticSolution {
  using analytic_solution = void;
};
template <>
struct AnalyticSolution<false> {};

static_assert(evolution::has_analytic_solution_alias_v<AnalyticSolution<true>>,
              "Failed testing evolution::has_analytic_solution");
static_assert(
    not evolution::has_analytic_solution_alias_v<AnalyticSolution<false>>,
    "Failed testing evolution::has_analytic_solution");
}  // namespace
