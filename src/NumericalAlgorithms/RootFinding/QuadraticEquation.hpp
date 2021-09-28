// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares functions for solving quadratic equations

#pragma once

#include <array>

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the positive root of a quadratic equation \f$ax^2 +
 * bx + c = 0\f$
 * \returns The positive root of a quadratic equation.
 * \requires That there are two real roots, of which only one is positive.
 */
double positive_root(double a, double b, double c);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the smallest root of a quadratic equation \f$ax^2 +
 * bx + c = 0\f$ that is greater than the given value, within roundoff.
 * \returns A root of a quadratic equation.
 * \requires That there are two real roots.
 * \requires At least one root is greater than the given value, to roundoff.
 */
template <typename T>
T smallest_root_greater_than_value_within_roundoff(const T& a, const T& b,
                                                   const T& c, double value);
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the largest root of a quadratic equation
 * \f$ax^2 + bx + c = 0\f$ that is between min_value and max_value,
 * within roundoff.
 * \returns A root of a quadratic equation.
 * \requires That there are two real roots.
 * \requires At least one root is between min_value and max_value, to roundoff.
 */
template <typename T>
T largest_root_between_values_within_roundoff(const T& a, const T& b,
                                              const T& c, double min_value,
                                              double max_value);
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the two real roots of a quadratic equation \f$ax^2 +
 * bx + c = 0\f$ with the root closer to \f$-\infty\f$ first.
 * \returns An array of the roots of a quadratic equation
 * \requires That there are two real roots.
 */
std::array<double, 2> real_roots(double a, double b, double c);
