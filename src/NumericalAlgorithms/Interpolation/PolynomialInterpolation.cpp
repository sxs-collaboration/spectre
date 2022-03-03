// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"

#include <cstddef>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {
template <size_t Degree>
void polynomial_interpolation(const gsl::not_null<double*> y,
                              const gsl::not_null<double*> error_in_y,
                              const double target_x,
                              const gsl::span<const double>& y_values,
                              const gsl::span<const double>& x_values) {
  constexpr size_t order = Degree + 1;
  ASSERT(x_values.size() == order, "The x_values span must be of size "
                                       << order << " but got "
                                       << x_values.size());
  ASSERT(y_values.size() == order, "The x_values span must be of size "
                                       << order << " but got "
                                       << x_values.size());
  using std::abs;
  // C and D are constants used in the interpolation algorithm. They're an
  // intermediate variable to make life easier.
  std::array<double, order> algorithm_c{};
  std::array<double, order> algorithm_d{};
  double minimum_dx = abs(target_x - x_values[0]);
  size_t index_of_closest_point = 0;
  for (size_t i = 0; i < order; ++i) {
    const double local_dx = abs(target_x - x_values[i]);
    if (local_dx < minimum_dx) {
      index_of_closest_point = i;
      minimum_dx = local_dx;
    }
    gsl::at(algorithm_c, i) = y_values[i];
    gsl::at(algorithm_d, i) = y_values[i];
  }

  size_t ns = index_of_closest_point;
  *y = y_values[ns--];
  for (size_t m = 1; m < order; m++) {
    for (size_t i = 0; i < order - m; i++) {
      const double delta_x_o = x_values[i] - target_x;
      const double delta_x_p = x_values[i + m] - target_x;
      const double c_minus_d =
          gsl::at(algorithm_c, i + 1) - gsl::at(algorithm_d, i);
      ASSERT(delta_x_o - delta_x_p != 0.0,
             "Encountered repeated grid points. i:" << i << " m:" << m
                                                    << " x:" << x_values[i]);
      const double adjustment_factor = c_minus_d / (delta_x_o - delta_x_p);
      gsl::at(algorithm_d, i) = delta_x_p * adjustment_factor;
      gsl::at(algorithm_c, i) = delta_x_o * adjustment_factor;
    }
    *error_in_y = 2 * (ns + 1) < (order - m) ? gsl::at(algorithm_c, ns + 1)
                                             : gsl::at(algorithm_d, ns--);
    *y += *error_in_y;
  }
}

#define GET_DEGREE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                     \
  template void polynomial_interpolation<GET_DEGREE(data)>(        \
      gsl::not_null<double*> y, gsl::not_null<double*> error_in_y, \
      double target_x, const gsl::span<const double>& y_values,    \
      const gsl::span<const double>& x_values);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3, 4, 5, 6, 7))

#undef INSTANTIATION
#undef GET_DEGREE
}  // namespace intrp
