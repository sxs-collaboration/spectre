// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"

/// Functions used in more than one cylindrical_endcap map
namespace cylindrical_endcap_helpers {
/// @{
/// Returns \f$\sin(ax)/x\f$
double sin_ax_over_x(const double x, const double a);
DataVector sin_ax_over_x(const DataVector& x, const double a);
/// @}

/// @{
/// Returns \f$\frac{1}{x} \frac{d}{dx}\left( \frac{\sin(ax)}{x} \right)\f$,
/// which approaches a finite limit as \f$x\f$ approaches zero.
double one_over_x_d_sin_ax_over_x(const double x, const double a);
DataVector one_over_x_d_sin_ax_over_x(const DataVector& x,
                                      const double a);
/// @}
}  // namespace cylindrical_endcap_helpers
