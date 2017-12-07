// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function find_root_of_function

#pragma once

#include <functional>
#include <limits>

#include <boost/math/tools/toms748_solve.hpp>

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`. An example is below.
 *
 * \snippet Test_OneDRootFinder.cpp double_root_find
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function f is invokable with a `double`
 */
template <typename Function>
double find_root_of_function(const Function& f, const double lower_bound,
                             const double upper_bound,
                             const double absolute_tolerance,
                             const double relative_tolerance,
                             const size_t max_iterations = 100) {
  boost::uintmax_t max_iter = max_iterations;

  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  auto tol = [absolute_tolerance, relative_tolerance](double lhs, double rhs) {
    return (fabs(lhs - rhs) <=
            absolute_tolerance +
                relative_tolerance * fmin(fabs(lhs), fabs(rhs)));
  };
  // Lower and upper bound are shifted by absolute tolerance so that the root
  // find does not fail if upper or lower bound are equal to the root within
  // tolerance
  auto result = boost::math::tools::toms748_solve(
      f, lower_bound - absolute_tolerance, upper_bound + absolute_tolerance,
      tol, max_iter);
  return result.first + 0.5 * (result.second - result.first);
}
