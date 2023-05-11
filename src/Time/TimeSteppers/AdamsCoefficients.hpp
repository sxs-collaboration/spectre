// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container/static_vector.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>

#include "Time/Time.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"

/// Helpers for calculating Adams coefficients
namespace TimeSteppers::adams_coefficients {
constexpr size_t maximum_order = 8;

/// A vector holding one entry per order of integration.
template <typename T>
using OrderVector = boost::container::static_vector<T, maximum_order>;

/// The standard Adams-Bashforth coefficients for constant step size,
/// as one would find in a reference table.
OrderVector<double> constant_adams_bashforth_coefficients(size_t order);

/// \brief Generate coefficients for an Adams step.
///
/// The coefficients are for a step using derivatives at \p
/// control_times.  The step is taken from the last time in \p
/// control_times to 0.  The result includes the overall factor of
/// step size, so, for example, the coefficients for Euler's method
/// (`control_times = {-dt}`) would be `{dt}`, not `{1}`.
///
/// No requirements are imposed on \p control_times, except that the
/// entries are all distinct.
///
/// Only `T = double` is used by the time steppers, but `T = Rational`
/// can be used to generate coefficient tables.
template <typename T>
OrderVector<T> variable_coefficients(const OrderVector<T>& control_times);

/// \brief Get coefficients for a time step.
///
/// Arguments are an iterator pair to past times (of type `Time`),
/// with the most recent last, and the time step to take as a type
/// with a `value()` method returning `double`.  The returned
/// coefficients include the factor of the step size, so, for example,
/// the coefficients for Euler's method would be `{step.value()}`, not
/// `{1}`.
template <typename Iterator, typename Delta>
OrderVector<double> coefficients(const Iterator& times_begin,
                                 const Iterator& times_end, const Delta& step) {
  bool constant_step_size = true;
  OrderVector<double> control_times;
  for (auto t = times_begin; t != times_end; ++t) {
    // Ideally we would also include the slab size in the scale of the
    // roundoff comparison, but there's no good way to get it here,
    // and it should only matter for slabs near time zero.
    if (constant_step_size and not control_times.empty() and
        not equal_within_roundoff(
            t->value() - control_times.back(), step.value(),
            100.0 * std::numeric_limits<double>::epsilon(), abs(t->value()))) {
      constant_step_size = false;
    }
    control_times.push_back(t->value());
  }
  if (constant_step_size) {
    auto result = constant_adams_bashforth_coefficients(control_times.size());
    alg::for_each(result, [&](double& coef) { coef *= step.value(); });
    return result;
  }

  const double goal_time = control_times.back() + step.value();
  for (auto& t : control_times) {
    t -= goal_time;
  }

  return variable_coefficients(control_times);
}
}  // namespace TimeSteppers::adams_coefficients
