// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the spatial part of the Z4 constraint
 *
 * \details Computes the constraint as:
 *
 * \f{align}
 *     Z_i &= \frac{1}{2} \tilde{\gamma}_{ij} \left(
 *         \hat{\Gamma}^j - \tilde{\Gamma}^j\right)
 * \f}
 *
 * where \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric defined by
 * `Ccz4::Tags::ConformalMetric` and
 * \f$\left(\hat{\Gamma}^i - \tilde{\Gamma}^i\right)\f$ is the CCZ4 temporary
 * expression defined by
 * `Ccz4::Tags::GammaHatMinusContractedConformalChristoffel`.
 */
template <size_t Dim, typename Frame, typename DataType>
void spatial_z4_constraint(
    const gsl::not_null<tnsr::i<DataType, Dim, Frame>*> result,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel);

template <size_t Dim, typename Frame, typename DataType>
tnsr::i<DataType, Dim, Frame> spatial_z4_constraint(
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel);
/// @}
}  // namespace Ccz4
