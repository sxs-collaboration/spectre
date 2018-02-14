// Distributed under the MIT License.
// See LICENSE.txt for details.

/// Defines functions useful for testing StrahlkorperGr.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "tests/Unit/TestHelpers.hpp"

/*!
 * \brief Schwarzschild (Kerr-Schild) spatial ricci tensor
 *
 * \details
 * Computes \f$R_{ij} = M \frac{r^2(4M+r)\delta_{ij}-(8M+3r)x_i x_j}
 * {r^4(2M+r^2)},\f$
 * where \f$r = x_i x_j \delta^{ij}\f$, \f$x_i\f$ is the
 * position vector in Cartesian coordinates, and M is the mass.
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> make_spatial_ricci_schwarzschild(
    const tnsr::A<DataType, SpatialDim, Frame, Index>& x,
    const double& mass) noexcept;
