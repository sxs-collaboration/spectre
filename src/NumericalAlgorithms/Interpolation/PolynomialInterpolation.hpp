// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/Gsl.hpp"

namespace intrp {
/*!
 * \brief Interpolate `y_values` to `target_x` from tabulated `x_values` using a
 * polynomial interpolant of degree `Degree`.
 *
 * `error_in_y` is an estimate of the error of the interpolated value. Note that
 * at least in the tests this is a significant overestimate of the errors
 * (several orders of magnitude). However, this could be because in the test the
 * polynomial can be represented exactly when all terms are present, but incurs
 * significant errors when the largest degree term is omitted.
 */
template <size_t Degree>
void polynomial_interpolation(gsl::not_null<double*> y,
                              gsl::not_null<double*> error_in_y,
                              double target_x,
                              const gsl::span<const double>& y_values,
                              const gsl::span<const double>& x_values);
}  // namespace intrp
