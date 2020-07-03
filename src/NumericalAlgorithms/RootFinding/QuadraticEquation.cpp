// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

#include <gsl/gsl_poly.h>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"

double positive_root(const double a, const double b, const double c) noexcept {
  const auto roots = real_roots(a, b, c);
  ASSERT(roots[0] <= 0.0 and roots[1] >= 0.0,
         "There are two positive roots, " << roots[0] << " and " << roots[1]
         << ", with a=" << a << " b=" << b << " c=" << c);
  return roots[1];
}

std::array<double, 2> real_roots(const double a, const double b,
                                 const double c) noexcept {
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

template <typename T>
struct smallest_root_greater_than_value_within_roundoff_impl;

template <typename T>
T smallest_root_greater_than_value_within_roundoff(const T& a, const T& b,
                                                   const T& c,
                                                   double value) noexcept {
  return smallest_root_greater_than_value_within_roundoff_impl<T>::f(a, b, c,
                                                                     value);
}

template <>
struct smallest_root_greater_than_value_within_roundoff_impl<double> {
  static double f(const double a, const double b, const double c,
                  const double value) noexcept {
    const auto roots = real_roots(a, b, c);
    // Roots are returned in increasing order.

    if (roots[0] >= value or equal_within_roundoff(roots[0], value)) {
      return roots[0];
    }
    ASSERT(roots[1] >= value or equal_within_roundoff(roots[1], value),
           "No root >=1.  Roots are " << roots[0] << " and " << roots[1]
                                      << ", with a=" << a << " b=" << b
                                      << " c=" << c);
    return roots[1];
  }
};

template <>
struct smallest_root_greater_than_value_within_roundoff_impl<DataVector> {
  static DataVector f(const DataVector& a, const DataVector& b,
                      const DataVector& c, const double value) noexcept {
    ASSERT(a.size() == b.size(),
           "Size mismatch a vs b: " << a.size() << " " << b.size());
    ASSERT(a.size() == c.size(),
           "Size mismatch a vs c: " << a.size() << " " << c.size());
    DataVector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      result[i] = smallest_root_greater_than_value_within_roundoff(a[i], b[i],
                                                                   c[i], value);
    }
    return result;
  }
};

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template DTYPE(data) smallest_root_greater_than_value_within_roundoff(   \
      const DTYPE(data) & a, const DTYPE(data) & b, const DTYPE(data) & c, \
      double value) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
