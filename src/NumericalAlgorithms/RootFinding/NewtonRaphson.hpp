// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function RootFinder::newton_raphson

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/MakeString.hpp"

namespace RootFinder {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the Newton-Raphson
 * method, falling back to bisection on poor convergence.
 *
 * `f` is a unary invokable that takes a `double` which is the current
 * value at which to evaluate `f`. `f` must return a
 * `std::pair<double, double>` where the first element is the function
 * value and the second element is the derivative of the function.
 * The method converges when the residual is smaller than or equal to
 * \p residual_tolerance or when the step size is smaller than \p
 * step_absolute_tolerance or the proposed result times \p
 * step_relative_tolerance.
 *
 * \snippet Test_NewtonRaphson.cpp double_newton_raphson_root_find
 *
 * \requires Function `f` is invokable with a `double`
 *
 * \throws `convergence_error` if the requested precision is not met after
 * `max_iterations` iterations.
 */
template <typename Function>
double newton_raphson(const Function& f, const double initial_guess,
                      const double lower_bound, const double upper_bound,
                      const double residual_tolerance,
                      const double step_absolute_tolerance,
                      const double step_relative_tolerance,
                      const size_t max_iterations = 50) {
  // Check if a and b have the same sign.  Zero does not have the same
  // sign as anything.
  const auto same_sign = [](const double a, const double b) {
    return a * b > 0.0;
  };

  ASSERT(residual_tolerance >= 0.0, "residual_tolerance must be non-negative.");
  ASSERT(step_absolute_tolerance >= 0.0,
         "step_absolute_tolerance must be non-negative.");
  ASSERT(step_relative_tolerance >= 0.0,
         "step_relative_tolerance must be non-negative.");

  ASSERT(not same_sign(f(lower_bound).first, f(upper_bound).first) or
         std::abs(f(upper_bound).first) < residual_tolerance or
         std::abs(f(lower_bound).first) < residual_tolerance,
         "Root not bracketed: "
         "f(" << lower_bound << ") = " << f(lower_bound).first << "  "
         "f(" << upper_bound << ") = " << f(upper_bound).first);

  // Avoid evaluating at the bounds if we do not need to.  This will
  // result in a non-optimal bracket until the region containing the
  // root has been determined, but will avoid extra function
  // evaluations if bisection is never needed.
  std::optional<double> bracket_positive{};
  std::optional<double> bracket_negative{};

  double x = initial_guess;
  double step = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < max_iterations; ++i) {
    const auto [value, deriv] = f(x);
    if (std::abs(value) <= residual_tolerance) {
      return x;
    }
    if (value > 0.0) {
      bracket_positive.emplace(x);
    } else {
      bracket_negative.emplace(x);
    }

    const double previous_x = x;
    const auto current_bracket =
        bracket_positive.has_value() and bracket_negative.has_value()
            ? std::pair(*bracket_positive, *bracket_negative)
            : std::pair(upper_bound, lower_bound);
    if (same_sign((x - current_bracket.first) * deriv - value,
                  (x - current_bracket.second) * deriv - value) or
        std::abs(2.0 * value) > std::abs(step * deriv)) {
      // Next guess not bracketed or converging slowly.  Perform a
      // bisection.

      // We need a bracket to bisect, so find any unknown bounds.
      if (not(bracket_positive.has_value() and bracket_negative.has_value())) {
        const double low_value = f(lower_bound).first;
        if (std::abs(low_value) <= residual_tolerance) {
          return lower_bound;
        }

        // Only one can be unset, because we set one of them right
        // after we evaluated f(x).
        if (not bracket_positive.has_value()) {
          bracket_positive.emplace(low_value > 0.0 ? lower_bound : upper_bound);
        } else {
          bracket_negative.emplace(low_value > 0.0 ? upper_bound : lower_bound);
        }
      }
      x = 0.5 * (*bracket_positive + *bracket_negative);
    } else {
      // Convergence is fine.  Keep going with plain Newton-Raphson.
      x -= value / deriv;
    }
    // Roundoff effects may make this different from value/deriv in
    // the Newton-Raphson case.
    step = std::abs(x - previous_x);
    if (step <= step_absolute_tolerance or
        step <= step_relative_tolerance * std::abs(x)) {
      return x;
    }
  }

  throw convergence_error(MakeString{}
                          << "newton_raphson reached max iterations of "
                          << max_iterations
                          << " without converging. Best result is: " << x
                          << " with residual " << f(x).first);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the Newton-Raphson method on
 * each element in a `DataVector`.
 *
 * Calls the `double` version of `newton_raphson` for each element of
 * the input `DataVector`s, passing an additional `size_t` to \p f
 * indicating the current index into the `DataVector`s.
 *
 * \snippet Test_NewtonRaphson.cpp datavector_newton_raphson_root_find
 *
 * \requires Function `f` be callable with a `double` and a `size_t`
 *
 * \throws `convergence_error` if, for any index, the requested precision is not
 * met after `max_iterations` iterations.
 */
template <typename Function>
DataVector newton_raphson(const Function& f, const DataVector& initial_guess,
                          const DataVector& lower_bound,
                          const DataVector& upper_bound,
                          const double residual_tolerance,
                          const double step_absolute_tolerance,
                          const double step_relative_tolerance,
                          const size_t max_iterations = 50) {
  DataVector result_vector{lower_bound.size()};
  for (size_t i = 0; i < result_vector.size(); ++i) {
    result_vector[i] = newton_raphson(
        [&f, i](const double x) { return f(x, i); }, initial_guess[i],
        lower_bound[i], upper_bound[i], residual_tolerance,
        step_absolute_tolerance, step_relative_tolerance, max_iterations);
  }
  return result_vector;
}

}  // namespace RootFinder
