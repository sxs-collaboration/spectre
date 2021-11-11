// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the gradient of the spatial part of the Z4 constraint
 *
 * \details Computes the gradient as:
 *
 * \f{align}
 *     \nabla_i Z_j &=
 *         D_{ijl} \left(\hat{\Gamma}^l - \tilde{\Gamma}^l\right) +
 *         \frac{1}{2} \tilde{\gamma}_{jl} \left(
 *             \partial_i \hat{\Gamma}^l - \partial_i \tilde{\Gamma}^l\right) -
 *         \Gamma^l_{ij} Z_l
 * \f}
 *
 * where \f$Z_i\f$ is the spatial Z4 constraint defined by
 * `Ccz4::Tags::SpatialZ4Constraint`, \f$\tilde{\gamma}_{ij}\f$ is the conformal
 * spatial metric defined by `Ccz4::Tags::ConformalMetric`, \f$\Gamma^k_{ij}\f$
 * is the spatial Christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ChristoffelSecondKind`, \f$D_{ijk}\f$ is the CCZ4 auxiliary
 * variable defined by `Ccz4::Tags::FieldD`,
 * \f$\left(\hat{\Gamma}^i - \tilde{\Gamma}^i\right)\f$ is the CCZ4 temporary
 * expression defined by
 * `Ccz4::Tags::GammaHatMinusContractedConformalChristoffel`, and
 * \f$\left(\partial_i \hat{\Gamma}^j - \partial_i \tilde{\Gamma}^j\right)\f$ is
 * its spatial derivative.
 */
template <size_t Dim, typename Frame, typename DataType>
void grad_spatial_z4_constraint(
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> result,
    const tnsr::i<DataType, Dim, Frame>& spatial_z4_constraint,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel,
    const tnsr::iJ<DataType, Dim, Frame>&
        d_gamma_hat_minus_contracted_conformal_christoffel);

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> grad_spatial_z4_constraint(
    const tnsr::i<DataType, Dim, Frame>& spatial_z4_constraint,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel,
    const tnsr::iJ<DataType, Dim, Frame>&
        d_gamma_hat_minus_contracted_conformal_christoffel);
/// @}
}  // namespace Ccz4
