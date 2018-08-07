// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function RootFinder::toms748

#pragma once

#include <boost/math/tools/roots.hpp>
#include <functional>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Exceptions.hpp"

namespace RootFinder {

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`. An example is below.
 *
 * \snippet Test_TOMS748.cpp double_root_find
 *
 * The TOMS_748 algorithm searches for a root in the interval [`lower_bound`,
 * `upper_bound`], and will throw if this interval does not bracket a root,
 * i.e. if `f(lower_bound) * f(upper_bound) > 0`.
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` is invokable with a `double`
 *
 * \throws `std::domain_error` if the bounds do not bracket a root.
 * \throws `convergence_error` if the requested tolerance is not met after
 *                            `max_iterations` iterations.
 */
template <typename Function>
double toms748(const Function& f, const double lower_bound,
               const double upper_bound, const double absolute_tolerance,
               const double relative_tolerance,
               const size_t max_iterations = 100) {
  ASSERT(relative_tolerance > std::numeric_limits<double>::epsilon(),
         "The relative tolerance is too small.");

  boost::uintmax_t max_iters = max_iterations;

  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  auto tol = [absolute_tolerance, relative_tolerance](double lhs, double rhs) {
    return (fabs(lhs - rhs) <=
            absolute_tolerance +
                relative_tolerance * fmin(fabs(lhs), fabs(rhs)));
  };
  // clang-tidy: internal boost warning, can't fix it.
  auto result = boost::math::tools::toms748_solve(  // NOLINT
      f, lower_bound, upper_bound, tol, max_iters);
  if (max_iters >= max_iterations) {
    throw convergence_error(
        "toms748 reached max iterations without converging");
  }
  return result.first + 0.5 * (result.second - result.first);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method on each
 * element in a `DataVector`.
 *
 * `f` is a binary invokable that takes a `double` as its first argument and a
 * `size_t` as its second. The `double` is the current value at which to
 * evaluate `f`, and the `size_t` is the current index into the `DataVector`s.
 * Below is an example of how to root find different functions by indexing into
 * a lambda-captured `DataVector` using the `size_t` passed to `f`.
 *
 * \snippet Test_TOMS748.cpp datavector_root_find
 *
 * For each index `i` into the DataVector, the TOMS_748 algorithm searches for a
 * root in the interval [`lower_bound[i]`, `upper_bound[i]`], and will throw if
 * this interval does not bracket a root,
 * i.e. if `f(lower_bound[i], i) * f(upper_bound[i], i) > 0`.
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` be callable with a `double` and a `size_t`
 *
 * \throws `std::domain_error` if, for any index, the bounds do not bracket a
 * root.
 * \throws `convergence_error` if, for any index, the requested tolerance is not
 * met after `max_iterations` iterations.
 */
template <typename Function>
DataVector toms748(const Function& f, const DataVector& lower_bound,
                   const DataVector& upper_bound,
                   const double absolute_tolerance,
                   const double relative_tolerance,
                   const size_t max_iterations = 100) {
  ASSERT(relative_tolerance > std::numeric_limits<double>::epsilon(),
         "The relative tolerance is too small.");
  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  auto tol = [absolute_tolerance, relative_tolerance](const double lhs,
                                                      const double rhs) {
    return (fabs(lhs - rhs) <=
            absolute_tolerance +
                relative_tolerance * fmin(fabs(lhs), fabs(rhs)));
  };
  DataVector result_vector{lower_bound.size()};
  for (size_t i = 0; i < result_vector.size(); ++i) {
    // toms748_solver modifies the max_iters after the root is found to the
    // number of iterations that it took to find the root, so we reset it to
    // max_iterations after each root find.
    boost::uintmax_t max_iters = max_iterations;
    // clang-tidy: internal boost warning, can't fix it.
    auto result = boost::math::tools::toms748_solve(  // NOLINT
        [&f, i](double x) { return f(x, i); }, lower_bound[i], upper_bound[i],
        tol, max_iters);
    if (max_iters >= max_iterations) {
      throw convergence_error(
          "toms748 reached max iterations without converging");
    }
    result_vector[i] = result.first + 0.5 * (result.second - result.first);
  }
  return result_vector;
}

}  // namespace RootFinder
