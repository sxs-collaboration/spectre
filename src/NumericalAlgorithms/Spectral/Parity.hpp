// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdint>
#include <ostream>

namespace Spectral {
/*!
 * \brief Used to label parity, either Even or Odd.
 */
enum class Parity : std::uint8_t { Uninitialized, Even, Odd };

/// All possible values of Parity
std::array<Parity, 3> all_parities();

// Output operator for a Basis
std::ostream& operator<<(std::ostream& os, const Parity& parity);
}  // namespace Spectral
