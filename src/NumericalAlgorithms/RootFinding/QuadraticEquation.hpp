// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares functions for solving quadratic equations

#pragma once

#include <array>

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the positive root of a quadratic equation ax^2 + bx + c = 0
 * \returns The positive root of a quadratic equation.
 * \requires That there are two real roots, of which only one is positive.
 */
double positive_root(double a, double b, double c);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the two real roots of a quadratic equation ax^2 + bx + c =
 * 0
 * \returns An array of the roots of a quadratic equation
 * \requires That there are two real roots.
 */
std::array<double, 2> real_roots(double a, double b, double c);
