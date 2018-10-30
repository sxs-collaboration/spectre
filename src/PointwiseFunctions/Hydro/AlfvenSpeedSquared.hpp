// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace hydro {
/*!
 * \brief Alfvén speed squared from comoving magnetic field.
 *
 * \f{align*}
 * v_A^2 = \dfrac{b^2}{b^2 + \rho h}
 * \f}
 *
 * where \f$b^2 = b_\mu b^\mu\f$ is the comoving magnetic field squared,
 * \f$\rho\f$ is the rest mass density, and \f$h\f$ is the specific enthalpy.
 */
template <typename DataType>
Scalar<DataType> alfven_speed_squared(
    const Scalar<DataType>& comoving_magnetic_field_squared,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) noexcept;
}  // namespace hydro
