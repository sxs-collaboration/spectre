// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \ingroup UtilitiesGroup
/// Check whether the argument is a NaN.  This performs the same test
/// as std::isnan, but never raises a floating point exception.
/// @{
bool is_nan(double arg);
bool is_nan(float arg);
/// @}
