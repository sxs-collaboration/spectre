// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

#include <gsl/gsl_poly.h>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

double positive_root(const double a, const double b, const double c) {
  const auto roots = real_roots(a, b, c);
  ASSERT(roots[0] <= 0.0 and roots[1] >= 0.0,
         "There are two positive roots, " << roots[0] << " and " << roots[1]
         << ", with a=" << a << " b=" << b << " c=" << c);
  return roots[1];
}

std::array<double, 2> real_roots(const double a, const double b,
                                 const double c) {
  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  // clang-tidy: value stored ... never read (true if in Release Build)
  // NOLINTNEXTLINE
  const int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  ASSERT(num_real_roots == 2,
         "There are only " << num_real_roots << " real roots with a=" << a
         << " b=" << b << " c=" << c);
  return {{x0, x1}};
}

namespace detail {
template <typename T>
struct root_between_values_impl;
enum class RootToChoose { min, max };
}  // namespace detail

template <typename T>
T smallest_root_greater_than_value_within_roundoff(const T& a, const T& b,
                                                   const T& c,
                                                   const double value) {
  return detail::root_between_values_impl<T>::function(
      a, b, c, value, std::numeric_limits<double>::max(),
      detail::RootToChoose::min);
}

template <typename T>
T largest_root_between_values_within_roundoff(const T& a, const T& b,
                                              const T& c,
                                              const double min_value,
                                              const double max_value) {
  return detail::root_between_values_impl<T>::function(
      a, b, c, min_value, max_value, detail::RootToChoose::max);
}

namespace detail {
template <>
struct root_between_values_impl<double> {
  static double function(const double a, const double b, const double c,
                         const double min_value, const double max_value,
                         const RootToChoose min_or_max) {
    // Roots are returned in increasing order.
    const auto roots = real_roots(a, b, c);

    const std::array<bool, 2> roots_out_of_bounds_low{
        {roots[0] < min_value and
             not equal_within_roundoff(roots[0], min_value),
         roots[1] < min_value and
             not equal_within_roundoff(roots[1], min_value)}};
    const std::array<bool, 2> roots_out_of_bounds_high{
        {roots[0] > max_value and
             not equal_within_roundoff(roots[0], max_value),
         roots[1] > max_value and
             not equal_within_roundoff(roots[1], max_value)}};

    double return_value = std::numeric_limits<double>::signaling_NaN();
    const auto error_message = [&a, &b, &c,
                                &roots](const std::string& message) {
      ERROR(message << " Roots are " << roots[0] << " and " << roots[1]
                    << ", with a=" << a << " b=" << b << " c=" << c);
    };

    if (min_or_max == RootToChoose::min) {
      // Check roots[0] first because it is the smallest
      if (roots_out_of_bounds_low[0]) {
        if (roots_out_of_bounds_low[1]) {
          error_message("No root >= (within roundoff) min_value.");
        }
        if (roots_out_of_bounds_high[1]) {
          error_message(
              "No root between min_value and max_value (within roundoff).");
        }
        return_value = roots[1];
      } else {
        if (roots_out_of_bounds_high[0]) {
          error_message("No root <= (within roundoff) max_value.");
        }
        return_value = roots[0];
      }
    } else {
      // Check roots[1] first because it is the largest
      if (roots_out_of_bounds_high[1]) {
        if (roots_out_of_bounds_high[0]) {
          error_message("No root <= (within roundoff) max_value.");
        }
        if (roots_out_of_bounds_low[0]) {
          error_message(
              "No root between min_value and max_value (within "
              "roundoff).");
        }
        return_value = roots[0];
      } else {
        if (roots_out_of_bounds_low[1]) {
          error_message("No root >= (within roundoff) min_value.");
        }
        return_value = roots[1];
      }
    }
    return return_value;
  }
};

template <>
struct root_between_values_impl<DataVector> {
  static DataVector function(const DataVector& a, const DataVector& b,
                             const DataVector& c, const double min_value,
                             const double max_value,
                             const RootToChoose min_or_max) {
    ASSERT(a.size() == b.size(),
           "Size mismatch a vs b: " << a.size() << " " << b.size());
    ASSERT(a.size() == c.size(),
           "Size mismatch a vs c: " << a.size() << " " << c.size());
    DataVector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      result[i] = root_between_values_impl<double>::function(
          a[i], b[i], c[i], min_value, max_value, min_or_max);
    }
    return result;
  }
};
}  // namespace detail

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template DTYPE(data) smallest_root_greater_than_value_within_roundoff(   \
      const DTYPE(data) & a, const DTYPE(data) & b, const DTYPE(data) & c, \
      double value);                                                       \
  template DTYPE(data) largest_root_between_values_within_roundoff(        \
      const DTYPE(data) & a, const DTYPE(data) & b, const DTYPE(data) & c, \
      double min_value, double max_value);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
