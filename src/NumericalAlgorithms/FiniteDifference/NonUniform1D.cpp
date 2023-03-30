// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/NonUniform1D.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace fd {
template <size_t StencilSize>
std::array<std::array<double, StencilSize>, StencilSize> non_uniform_1d_weights(
    const std::deque<double>& times) {
  static_assert(StencilSize >= 2 and StencilSize <= 4,
                "Finite difference for a 1D non-uniform stencil is only "
                "implemented for stencil sizes 2, 3, and 4.");

  ASSERT(times.size() == StencilSize,
         "The size of the times passed in ("
             << times.size() << ") must be the same as the stencil size ("
             << StencilSize << ").");
  using ::operator<<;
  ASSERT(std::is_sorted(times.begin(), times.end(), std::greater<double>()),
         "Times must be monotonically decreasing: " << times);

  // initialize the finite difference coefs
  std::array<std::array<double, StencilSize>, StencilSize> coefs{};

  // These coefficients are the weights of the Lagrange interpolation
  // polynomial and its derivatives evaluated at `time[0]`
  if constexpr (StencilSize == 2) {
    const double one_over_delta_t = 1.0 / (times[0] - times[1]);

    // set coefs for function value
    coefs[0] = {{1.0, 0.0}};
    // first deriv coefs
    coefs[1] = {{one_over_delta_t, -one_over_delta_t}};
  } else if constexpr (StencilSize == 3) {
    const double t0_minus_t1 = times[0] - times[1];
    const double t1_minus_t2 = times[1] - times[2];
    const double t0_minus_t2 = times[0] - times[2];
    const double denom = 1.0 / t0_minus_t2;
    const double one_over_mult_dts = 1.0 / (t0_minus_t1 * t1_minus_t2);

    // set coefs for function value
    coefs[0] = {{1.0, 0.0, 0.0}};
    // first deriv coefs
    coefs[1] = {{(2.0 + t1_minus_t2 / t0_minus_t1) * denom,
                 -t0_minus_t2 * one_over_mult_dts,
                 t0_minus_t1 / t1_minus_t2 * denom}};
    // second deriv coefs
    coefs[2] = {{2.0 / t0_minus_t1 * denom, -2.0 * one_over_mult_dts,
                 2.0 / t1_minus_t2 * denom}};
  } else if constexpr (StencilSize == 4) {
    const double t1_minus_t0 = times[1] - times[0];
    const double t2_minus_t0 = times[2] - times[0];
    const double t3_minus_t0 = times[3] - times[0];
    const double t1_minus_t2 = times[1] - times[2];
    const double t1_minus_t3 = times[1] - times[3];
    const double t2_minus_t3 = times[2] - times[3];

    // set coefs for function value
    coefs[0] = {{1.0, 0.0, 0.0, 0.0}};
    // first deriv coefs
    coefs[1] = {
        {-1.0 / t1_minus_t0 - 1.0 / t2_minus_t0 - 1.0 / t3_minus_t0,
         t2_minus_t0 * t3_minus_t0 / (t1_minus_t0 * t1_minus_t2 * t1_minus_t3),
         -t1_minus_t0 * t3_minus_t0 / (t2_minus_t0 * t1_minus_t2 * t2_minus_t3),
         t1_minus_t0 * t2_minus_t0 /
             (t3_minus_t0 * t1_minus_t3 * t2_minus_t3)}};
    // second deriv coefs
    coefs[2] = {{2.0 * (t1_minus_t0 + t2_minus_t0 + t3_minus_t0) /
                     (t1_minus_t0 * t2_minus_t0 * t3_minus_t0),
                 -2.0 * (t2_minus_t0 + t3_minus_t0) /
                     (t1_minus_t0 * t1_minus_t2 * t1_minus_t3),
                 2.0 * (t1_minus_t0 + t3_minus_t0) /
                     (t2_minus_t0 * t1_minus_t2 * t2_minus_t3),
                 -2.0 * (t1_minus_t0 + t2_minus_t0) /
                     (t3_minus_t0 * t1_minus_t3 * t2_minus_t3)}};
    // third deriv coefs
    coefs[3] = {{-6.0 / (t1_minus_t0 * t2_minus_t0 * t3_minus_t0),
                 6.0 / (t1_minus_t0 * t1_minus_t2 * t1_minus_t3),
                 -6.0 / (t2_minus_t0 * t1_minus_t2 * t2_minus_t3),
                 6.0 / (t3_minus_t0 * t1_minus_t3 * t2_minus_t3)}};
  }

  return coefs;
}

#define STENCIL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                          \
  template std::array<std::array<double, STENCIL(data)>, STENCIL(data)> \
  non_uniform_1d_weights(const std::deque<double>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3, 4))

#undef STENCIL
#undef INSTANTIATION
}  // namespace fd
