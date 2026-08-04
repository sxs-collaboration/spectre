// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Hash.hpp"

#include <boost/functional/hash.hpp>
#include <functional>
#include <utility>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<std::pair<Spectral::Basis, Spectral::Quadrature>>::operator()(
    const std::pair<Spectral::Basis, Spectral::Quadrature>& p) const {
  return boost::hash<std::pair<Spectral::Basis, Spectral::Quadrature>>{}(p);
}
}  // namespace std
