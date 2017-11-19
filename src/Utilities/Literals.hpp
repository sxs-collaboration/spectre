// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines useful literals

#pragma once

#include <string>

using namespace std::literals::string_literals;  // NOLINT

/// \ingroup UtilitiesGroup
/// Defines the _st size_t suffix
inline constexpr size_t operator"" _st(const unsigned long long n) {  // NOLINT
  return static_cast<size_t>(n);
}
