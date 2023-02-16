// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>

/// Check whether an object contains memory allocations.  Classes can
/// add overloads as needed.
/// @{
inline bool contains_allocations(const double /*value*/) { return false; }
inline bool contains_allocations(const std::complex<double> /*value*/) {
  return false;
}
/// @}
