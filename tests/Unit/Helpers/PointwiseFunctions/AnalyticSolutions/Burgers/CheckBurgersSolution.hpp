// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

template <typename Solution>
void check_burgers_solution(const Solution& solution,
                            const DataVector& positions,
                            const std::vector<double>& times) noexcept {
  static_assert(evolution::is_analytic_solution_v<Solution>,
                "Solution was not derived from AnalyticSolution");
  // Check that DataVector and double functions are consistent.
  const tnsr::I<DataVector, 1> positions_tnsr{{{positions}}};
  for (const double time : times) {
    const auto value = solution.u(positions_tnsr, time);
    CHECK(get<Burgers::Tags::U>(solution.variables(
              positions_tnsr, time, tmpl::list<Burgers::Tags::U>{})) == value);
    for (size_t point = 0; point < positions.size(); ++point) {
      const tnsr::I<double, 1> xp{{{positions[point]}}};
      const double up = get(value)[point];
      CHECK(get(solution.u(xp, time)) == approx(up));
    }
  }
}
