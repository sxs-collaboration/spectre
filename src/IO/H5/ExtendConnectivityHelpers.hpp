// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

/// \cond
namespace SpatialDiscretization {
enum class Basis;
enum class Quadrature;
}  // namespace SpatialDiscretization
/// \endcond

/// \cond
namespace h5::detail {
template <size_t SpatialDim>
std::vector<int> extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<SpatialDiscretization::Basis>>& bases,
    std::vector<std::vector<SpatialDiscretization::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents);
}

/// \endcond
