// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

/// @{
/*!
 * \brief The comoving magnetic field one-form $b_\mu$
 *
 * The components of the comoving magnetic field vector are:
 *
 * \begin{align}
 * b^0 &= W B^j v^k \gamma_{j k} / \alpha \\
 * b^i &= B^i / W + B^j v^k \gamma_{j k} u^i
 * \end{align}
 *
 * Using the spacetime metric, the corresponding one-form components are:
 *
 * \begin{align}
 * b_0 &= - \alpha W v^i B_i + \beta^i b_i \\
 * b_i &= B_i / W + B^j v^k \gamma_{j k} W v_i
 * \end{align}
 *
 * The square of the vector is:
 *
 * \begin{equation}
 * b^2 = B^i B^j \gamma_{i j} / W^2 + (B^i v^j \gamma_{i j})^2
 * \end{equation}
 *
 * See also Eq. (5.173) in \cite BaumgarteShapiro, with the difference that we
 * work in Heaviside-Lorentz units where the magnetic field is rescaled by
 * $1/\sqrt{4\pi}$, following \cite Moesta2013dna .
 */
template <typename DataType>
void comoving_magnetic_field_one_form(
    const gsl::not_null<tnsr::a<DataType, 3>*> result,
    const tnsr::i<DataType, 3>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const tnsr::I<DataType, 3>& shift,
    const Scalar<DataType>& lapse);

template <typename DataType>
tnsr::a<DataType, 3> comoving_magnetic_field_one_form(
    const tnsr::i<DataType, 3>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const tnsr::I<DataType, 3>& shift,
    const Scalar<DataType>& lapse);

template <typename DataType>
void comoving_magnetic_field_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor);

template <typename DataType>
Scalar<DataType> comoving_magnetic_field_squared(
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor);
/// @}

}  // namespace hydro
