// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function RootFinder::newton_raphson

#pragma once

#include <boost/math/tools/roots.hpp>
#include <functional>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Exceptions.hpp"

namespace RootFinder {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the Newton-Raphson method.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`. An example is below.
 *
 * \snippet Test_NewtonRaphson.cpp double_newton_raphson_root_find
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` is invokable with a `double`
 * \note The parameter `digits` specifies the precision of the result in its
 * desired number of base-10 digits.
 *
 * \throws `convergence_error` if the requested precision is not met after
 * `max_iterations` iterations.
 */
template <typename Function>
double newton_raphson(const Function& f, const double initial_guess,
                      const double lower_bound, const double upper_bound,
                      const size_t digits, const size_t max_iterations = 50) {
  ASSERT(digits < std::numeric_limits<double>::digits10,
         "The desired accuracy of " << digits
                                    << " base-10 digits must be smaller than "
                                       "the machine numeric limit of "
                                    << std::numeric_limits<double>::digits10
                                    << " base-10 digits.");

  boost::uintmax_t max_iters = max_iterations;
  // clang-tidy: internal boost warning, can't fix it.
  const auto result = boost::math::tools::newton_raphson_iterate(  // NOLINT
      f, initial_guess, lower_bound, upper_bound,
      std::round(std::log2(std::pow(10, digits))), max_iters);
  if (max_iters >= max_iterations) {
    throw convergence_error(
        "newton_raphson reached max iterations without converging");
  }
  return result;
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the Newton-Raphson method on
 * each element in a `DataVector`.
 *
 * `f` is a binary invokable that takes a `double` as its first argument and a
 * `size_t` as its second. The `double` is the current value at which to
 * evaluate `f`, and the `size_t` is the current index into the `DataVector`s.
 * Below is an example of how to root find different functions by indexing into
 * a lambda-captured `DataVector` using the `size_t` passed to `f`.
 *
 * \snippet Test_NewtonRaphson.cpp datavector_newton_raphson_root_find
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` be callable with a `double` and a `size_t`
 * \note The parameter `digits` specifies the precision of the result in its
 * desired number of base-10 digits.
 *
 * \throws `convergence_error` if, for any index, the requested precision is not
 * met after `max_iterations` iterations.
 */
template <typename Function>
DataVector newton_raphson(const Function& f, const DataVector& initial_guess,
                          const DataVector& lower_bound,
                          const DataVector& upper_bound, const size_t digits,
                          const size_t max_iterations = 50) {
  ASSERT(digits < std::numeric_limits<double>::digits10,
         "The desired accuracy of " << digits
                                    << " base-10 digits must be smaller than "
                                       "the machine numeric limit of "
                                    << std::numeric_limits<double>::digits10
                                    << " base-10 digits.");
  const auto digits_binary = std::round(std::log2(std::pow(10, digits)));

  DataVector result_vector{lower_bound.size()};
  for (size_t i = 0; i < result_vector.size(); ++i) {
    boost::uintmax_t max_iters = max_iterations;
    // clang-tidy: internal boost warning, can't fix it.
    result_vector[i] = boost::math::tools::newton_raphson_iterate(  // NOLINT
        [&f, i ](double x) noexcept { return f(x, i); }, initial_guess[i],
        lower_bound[i], upper_bound[i], digits_binary, max_iters);
    if (max_iters >= max_iterations) {
      throw convergence_error(
          "newton_raphson reached max iterations without converging");
    }
  }
  return result_vector;
}

}  // namespace RootFinder
