// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions lagrange_polynomial

#pragma once

#include <iterator>

#include "ErrorHandling/Assert.hpp"

/// Evaluate the jth Lagrange interpolating polynomial with the given
/// control points where j is the index of index_point.
template <typename Iterator>
double lagrange_polynomial(const Iterator& index_point,
                           double x,
                           const Iterator& control_points_begin,
                           const Iterator& control_points_end) {
  ASSERT(control_points_begin != control_points_end, "No control points");

  const double x_j = *index_point;
  double result = 1.;
  for (auto m_it = control_points_begin; m_it != control_points_end; ++m_it) {
    if (m_it == index_point) { continue; }
    const double x_m = *m_it;
    result *= (x_m - x) / (x_m - x_j);
  }
  return result;
}

/// Evaluate the jth (zero-indexed) Lagrange interpolating polynomial
/// with the given control points.
template <typename Iterator>
double lagrange_polynomial(size_t j, double x,
                           const Iterator& control_points_begin,
                           const Iterator& control_points_end) {
  const auto j_diff =
      static_cast<typename std::iterator_traits<Iterator>::difference_type>(j);
  ASSERT(j_diff < std::distance(control_points_begin, control_points_end),
         "Polynomial number out of range " << j << " > "
         << (std::distance(control_points_begin, control_points_end) - 1));
  return lagrange_polynomial(std::next(control_points_begin, j_diff),
                             x, control_points_begin, control_points_end);
}
