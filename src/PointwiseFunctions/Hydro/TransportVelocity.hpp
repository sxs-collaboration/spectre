// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace hydro {
/*!
 * \brief Transport velocity from spatial velocity, lapse and shift.
 *
 * \f{align*}
 * v_\text{tr}^i = \alpha v^i - \beta^i
 * \f}
 *
 * where \f$v^i\f$ is the spatial velocity, \f$\alpha\f$ is the lapse,
 * and \f$\beta^i\f$ is the shift.
 */
template <typename DataType, size_t Dim, typename Fr>
tnsr::I<DataType, Dim, Fr> transport_velocity(
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Fr>& shift) noexcept;
}  // namespace hydro
