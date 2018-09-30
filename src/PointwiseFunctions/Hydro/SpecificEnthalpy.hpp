// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace hydro {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Computes the relativistic specific enthalpy \f$h\f$ as:
 * \f$ h = 1 + \epsilon + \frac{p}{\rho} \f$
 * where \f$\epsilon\f$ is the specific internal energy, \f$p\f$
 * is the pressure, and \f$\rho\f$ is the rest mass density.
 */
template <typename DataType>
Scalar<DataType> specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept;
}  // namespace hydro
