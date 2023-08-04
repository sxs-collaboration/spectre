// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <vector>

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

/// \cond
namespace h5::detail {
template <size_t SpatialDim>
std::vector<int> extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<Spectral::Basis>>& bases,
    std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents);
}

/// \endcond
