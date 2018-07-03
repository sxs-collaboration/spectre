// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

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
  // Check that different functions are consistent.
  const tnsr::I<DataVector, 1> positions_tnsr{{{positions}}};
  for (const double time : times) {
    const auto value = solution.u(positions_tnsr, time);
    const auto time_deriv = solution.du_dt(positions_tnsr, time);
    CHECK(get<Burgers::Tags::U>(solution.variables(
              positions_tnsr, time, tmpl::list<Burgers::Tags::U>{})) == value);
    CHECK(get<Tags::dt<Burgers::Tags::U>>(solution.variables(
              positions_tnsr, time,
              tmpl::list<Tags::dt<Burgers::Tags::U>>{})) ==
          time_deriv);
    for (size_t point = 0; point < positions.size(); ++point) {
      const tnsr::I<double, 1> xp{{{positions[point]}}};
      const double up = get(value)[point];
      const double dtup = get(time_deriv)[point];
      CHECK(get(solution.u(xp, time)) == approx(up));
      CHECK(get(solution.du_dt(xp, time)) == approx(dtup));
      // Check that the time derivative is the derivative of the
      // value.
      CHECK(numerical_derivative(
                [&solution, &xp](const std::array<double, 1>& t) noexcept {
                  return std::array<double, 1>{{get(solution.u(xp, t[0]))}};
                },
                std::array<double, 1>{{time}}, 0, 1e-4)[0] ==
            approx.epsilon(1e-10)(dtup));
      // Check that the Burgers equation is satisfied.
      CHECK(numerical_derivative([&solution, &time](
                                     const std::array<double, 1>& x) noexcept {
              tnsr::I<DataVector, 1> flux{{{DataVector(1)}}};
              Burgers::Fluxes::apply(
                  &flux, solution.u(tnsr::I<DataVector, 1>{{{{x[0]}}}}, time));
              return std::array<double, 1>{{get<0>(flux)[0]}};
            },
                                 std::array<double, 1>{{get<0>(xp)}}, 0,
                                 1e-4)[0] == approx(-dtup));
    }
  }
}
