// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

/// \cond
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral

namespace h5::detail {
template <size_t SpatialDim>
std::vector<int> extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<Spectral::Basis>>& bases,
    std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents);
}

/// \endcond
