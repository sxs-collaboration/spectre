// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

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
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> spatial_ricci(
    const tnsr::I<DataType, SpatialDim, Frame>& x, double mass);
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
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature_sphere(
    const tnsr::I<DataType, SpatialDim, Frame>& x);
}  // namespace Minkowski

namespace Kerr {
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Kerr (Kerr-Schild) horizon ricci scalar (spin on z axis)
 *
 * \details
 * Computes the 2-dimensional Ricci scalar \f$R\f$ on the
 * horizon of a Kerr-Schild black hole with spin in the z direction
 * in terms of mass `mass` and dimensionless spin `dimensionless_spin_z`.
 */
template <typename DataType>
Scalar<DataType> horizon_ricci_scalar(const Scalar<DataType>& horizon_radius,
                                      double mass, double dimensionless_spin_z);

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Kerr (Kerr-Schild) horizon ricci scalar (generic spin)
 *
 * \details
 * Computes the 2-dimensional Ricci scalar \f$R\f$ on the
 * horizon of a Kerr-Schild black hole with generic spin
 * in terms of mass `mass` and dimensionless spin `dimensionless_spin`.
 */
template <typename DataType>
Scalar<DataType> horizon_ricci_scalar(
    const Scalar<DataType>& horizon_radius_with_spin_on_z_axis,
    const ylm::Spherepack& ylm_with_spin_on_z_axis, const ylm::Spherepack& ylm,
    double mass, const std::array<double, 3>& dimensionless_spin);

}  // namespace Kerr
}  // namespace TestHelpers
