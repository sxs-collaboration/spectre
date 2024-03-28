// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container/small_vector.hpp>
#include <cstddef>
#include <tuple>

#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"

/// \cond
namespace TimeSteppers {
template <typename T>
class BoundaryHistoryEvaluator;
}  // namespace TimeSteppers
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// Shared LTS implementation for the two Adams-based methods.
namespace TimeSteppers::adams_lts {
// For order-k 2:1 stepping, in each small step, half the points will
// require interpolation (k entries each) and the others will not (1
// entry each).  So (2 small steps) * ((k/2 interpolations) * (from k
// points) + (k/2 non-interpolations)) = k (k + 1).  (The small steps
// round k/2 in different directions and the effect cancels out.)
constexpr size_t lts_coefficients_static_size =
    adams_coefficients::maximum_order * (adams_coefficients::maximum_order + 1);

/// Storage for LTS coefficients that should not allocate in typical
/// cases.  Each entry is a tuple of (local id, remote id,
/// coefficient).  The contents should be kept sorted, as some
/// functions assume that.
struct LtsCoefficients
    : boost::container::small_vector<std::tuple<TimeStepId, TimeStepId, double>,
                                     lts_coefficients_static_size> {
  using boost::container::small_vector<
      std::tuple<TimeStepId, TimeStepId, double>,
      lts_coefficients_static_size>::small_vector;
};

LtsCoefficients& operator+=(LtsCoefficients& a, const LtsCoefficients& b);
LtsCoefficients& operator-=(LtsCoefficients& a, const LtsCoefficients& b);

LtsCoefficients operator+(LtsCoefficients&& a, LtsCoefficients&& b);
LtsCoefficients operator+(LtsCoefficients&& a, const LtsCoefficients& b);
LtsCoefficients operator+(const LtsCoefficients& a, LtsCoefficients&& b);
LtsCoefficients operator+(const LtsCoefficients& a, const LtsCoefficients& b);

LtsCoefficients operator-(LtsCoefficients&& a, LtsCoefficients&& b);
LtsCoefficients operator-(LtsCoefficients&& a, const LtsCoefficients& b);
LtsCoefficients operator-(const LtsCoefficients& a, LtsCoefficients&& b);
LtsCoefficients operator-(const LtsCoefficients& a, const LtsCoefficients& b);

/// Add the LTS boundary terms for to \p result for the given set of
/// coefficients.
template <typename T>
void apply_coefficients(gsl::not_null<T*> result,
                        const LtsCoefficients& coefficients,
                        const BoundaryHistoryEvaluator<T>& coupling);
}  // namespace TimeSteppers::adams_lts
