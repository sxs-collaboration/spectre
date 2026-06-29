// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Filters::detail {
// Returns true if `basis` is a Legendre or Chebyshev basis collocated on
// Gauss or Gauss-Lobatto points, i.e. a standard one-dimensional spectral
// direction usable by the exponential filter engine.
inline bool is_legendre_or_chebyshev(const Spectral::Basis basis,
                                     const Spectral::Quadrature quadrature) {
  return (basis == Spectral::Basis::Legendre or
          basis == Spectral::Basis::Chebyshev) and
         (quadrature == Spectral::Quadrature::Gauss or
          quadrature == Spectral::Quadrature::GaussLobatto);
}
}  // namespace Filters::detail
