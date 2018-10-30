// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace hydro {
/*!
 * \brief Comoving magnetic field from Eulerian magnetic field.
 *
 * \f{align*}
 * b^0 = \frac{WB^iv_i}{\alpha},\quad b^i = \frac{B^i}{W} + b^0 v_\text{tr}^i,
 * \f}
 *
 * where \f$B^i\f$ is the Eulerian magnetic field, \f$v_i\f$ is the spatial
 * velocity one-form, \f$v_\text{tr}^i = \alpha v^i - \beta^i\f$ is the
 * so-called transport velocity, \f$W\f$ is the Lorentz factor,
 * \f$\alpha\f$ is the lapse, and \f$\beta^i\f$ is the shift.
 */
template <typename DataType, size_t Dim, typename Fr>
tnsr::A<DataType, Dim, Fr> comoving_magnetic_field(
    const tnsr::I<DataType, Dim, Fr>& eulerian_b_field,
    const tnsr::I<DataType, Dim, Fr>& transport_velocity,
    const tnsr::i<DataType, Dim, Fr>& spatial_velocity_oneform,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& lapse) noexcept;

/*!
 * \brief Comoving magnetic field squared from Eulerian magnetic field.
 *
 * \f{align*}
 * b^2 = \frac{B^iB_i}{W^2} + (B^iv_i)^2,
 * \f}
 *
 * where \f$B^i\f$ is the Eulerian magnetic field, \f$v_i\f$ is the spatial
 * velocity one-form, and \f$W\f$ is the Lorentz factor.
 */
template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> comoving_magnetic_field_squared(
    const tnsr::I<DataType, Dim, Fr>& eulerian_b_field,
    const Scalar<DataType>& eulerian_b_field_squared,
    const tnsr::i<DataType, Dim, Fr>& spatial_velocity_oneform,
    const Scalar<DataType>& lorentz_factor) noexcept;
}  // namespace hydro
