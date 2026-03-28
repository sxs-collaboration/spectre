// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
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

template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>> extend_connectivity_by_block(
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents,
    const std::vector<std::vector<Spectral::Basis>>& block_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& block_quadratures);

template <size_t SpatialDim>
std::vector<int> new_extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<Spectral::Basis>>& bases,
    std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents);
}
/// \endcond
