// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container/static_vector.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>

#include "Time/Time.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/Algorithm.hpp"

/// Helpers for calculating Adams coefficients
namespace TimeSteppers::adams_coefficients {
constexpr size_t maximum_order = 8;

/// A vector holding one entry per order of integration.
template <typename T>
using OrderVector = boost::container::static_vector<T, maximum_order>;

/// The standard Adams-Bashforth coefficients for constant step size,
/// ordered from oldest to newest time, as one would find in a
/// reference table (except likely in the opposite order).
OrderVector<double> constant_adams_bashforth_coefficients(size_t order);

/// \brief Generate coefficients for an Adams step.
///
/// The coefficients are for a step using derivatives at \p
/// control_times, with the entries in the result vector corresponding
/// to the passed times in order.  The step is taken from the last
/// time in \p control_times to 0.  The result includes the overall
/// factor of step size, so, for example, the coefficients for Euler's
/// method (`control_times = {-dt}`) would be `{dt}`, not `{1}`.
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
/// coefficients correspond to the passed times in order.  They
/// include the factor of the step size, so, for example, the
/// coefficients for Euler's method would be `{step.value()}`, not
/// `{1}`.
template <typename Iterator, typename Delta>
OrderVector<double> coefficients(const Iterator& times_begin,
                                 const Iterator& times_end, const Delta& step) {
  if (times_begin == times_end) {
    return {};
  }
  const double step_size = step.value();
  bool constant_step_size = true;
  // We shift the control times to be near zero, which gives smaller
  // errors from the variable_coefficients function.
  OrderVector<double> control_times{0.0};
  Time previous_time = *times_begin;
  for (auto t = std::next(times_begin); t != times_end; ++t) {
    const Time this_time = *t;
    const double this_step = (this_time - previous_time).value();
    control_times.push_back(control_times.back() + this_step);
    if (constant_step_size and
        std::abs(this_step - step_size) > slab_rounding_error(this_time)) {
      constant_step_size = false;
    }
    previous_time = this_time;
  }
  if (constant_step_size) {
    auto result = constant_adams_bashforth_coefficients(control_times.size());
    alg::for_each(result, [&](double& coef) { coef *= step_size; });
    return result;
  }

  const double goal_time = control_times.back() + step_size;
  for (auto& t : control_times) {
    t -= goal_time;
  }

  return variable_coefficients(control_times);
}
}  // namespace TimeSteppers::adams_coefficients
