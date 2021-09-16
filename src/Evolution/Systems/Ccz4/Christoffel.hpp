// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conformal spatial christoffel symbols of the second kind.
 *
 * \details Computes the christoffel symbols as:
 * \f{align}
 *     \tilde{\Gamma}^k_{ij} &=
 *         \tilde{\gamma}^{kl} (D_{ijl} + D_{jil} - D_{lij})
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$ and \f$D_{ijk}\f$ are the inverse conformal
 * spatial metric and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::InverseConformalMetric` and `Ccz4::Tags::FieldD`, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::Ijj<DataType, Dim, Frame> conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) noexcept;
/// @}
}  // namespace Ccz4
