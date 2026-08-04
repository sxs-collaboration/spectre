// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>
#include <utility>

/// \cond
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace std {
template <>
// NOLINTNEXTLINE(cert-dcl58-cpp)
struct hash<std::pair<Spectral::Basis, Spectral::Quadrature>> {
  size_t operator()(
      const std::pair<Spectral::Basis, Spectral::Quadrature>& p) const;
};
}  // namespace std
