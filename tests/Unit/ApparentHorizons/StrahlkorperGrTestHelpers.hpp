// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace TestHelpers {
namespace Schwarzschild {
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Schwarzschild (Kerr-Schild) spatial ricci tensor
 *
 * \details
 * Computes \f$R_{ij} = M \frac{r^2(4M+r)\delta_{ij}-(8M+3r)x_i x_j}
 * {r^4(2M+r^2)},\f$
 * where \f$r = x_i x_j \delta^{ij}\f$, \f$x_i\f$ is the
 * position vector in Cartesian coordinates, and M is the mass.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_ricci(
    const tnsr::I<DataType, SpatialDim, Frame>& x, const double& mass) noexcept;
}  // namespace Schwarzschild

namespace Minkowski {
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Extrinsic curvature of 2D sphere in 3D flat space
 *
 * \details
 * Computes \f$K_{ij} = \frac{1}{r}\left(\delta_{ij} -
 * \frac{x_i x_j}{r}\right),\f$
 * where \f$r = x_i x_j \delta^{ij}\f$ and \f$x_i\f$ is the
 * position vector in Cartesian coordinates.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature_sphere(
    const tnsr::I<DataType, SpatialDim, Frame>& x) noexcept;
}  // namespace Minkowski

namespace Kerr {
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Radius of Kerr horizon in Kerr-Schild coordinates
 *
 * \details
 * Computes the radius of a Kerr black hole with mass `mass`
 * and dimensionless spin `spin`. The input
 * argument `theta_phi` is the output of the
 * `theta_phi_points()` method of a `YlmSpherepack` object;
 * i.e., it is typically a std::array of two DataVectors containing
 * the values of theta and phi at each point on a Strahlkorper.
 */
template <typename DataType>
Scalar<DataType> horizon_radius(const std::array<DataType, 2>& theta_phi,
                                const double& mass,
                                const std::array<double, 3>& spin) noexcept;

}  // namespace Kerr
}  // namespace TestHelpers
