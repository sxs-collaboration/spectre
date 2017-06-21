
/// \file
/// Declares function find_root_by_brents_method

#pragma once

#include <functional>
#include <limits>

#include <boost/math/tools/toms748_solve.hpp>


/*! \ingroup Functors
 *  \brief Finds the root of the function f with the TOMS_748 method.
 *
 *  \requires Function f be callable
 */
template <typename Function>
double find_root_of_function(Function f, const double lower_bound,
                             const double upper_bound,
                             const double absolute_tolerance,
                             const double relative_tolerance,
                             const size_t max_iterations = 100) {
  boost::uintmax_t max_iter = max_iterations;

  // This solver requires tol to be passed as a termination condition. This is
  // equivalent to the convergence criteria used by the GSL
  auto tol = [absolute_tolerance, relative_tolerance](double lhs, double rhs) {
    return (fabs(lhs - rhs) <=
            absolute_tolerance +
                relative_tolerance * fmin(fabs(lhs), fabs(rhs)));
  };
  auto result = boost::math::tools::toms748_solve(f, lower_bound, upper_bound,
                                                  tol, max_iter);
  return result.first + 0.5 * (result.second - result.first);
}
